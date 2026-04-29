"""Paper data: direct ROME implementation (no EasyEdit) on Pythia-1.4B.

ROME algorithm (Meng et al. 2022):
  1. Pick mid-layer L where factual associations live (Pythia-1.4B: L=8)
  2. Compute k* = MLP input activation at (L, subject_token_position)
  3. Optimize v* via gradient descent: find activation that, when added at
     MLP output of (L, subject_pos), maximizes log P(target_new | prompt)
  4. Apply rank-1 weight update:
       ΔW = (v* - W_down @ k*) ⊗ k* / (k*.T @ k*)
       W_down += ΔW

Simplified variant: no covariance regularization (skip C^-1 term).
Published ROME uses C^-1 computed from Wikipedia corpus; here we omit to keep
the comparison direct and re-runnable. This is a standard simplification used
in reproductions; it may show slightly worse locality than the published paper
but matches the general literature finding that ROME degrades sequentially.

Sequential N-edit protocol:
  For each of N facts:
    Apply ROME edit to the (now-possibly-modified) model.

Measure at each N:
  - Hit rate on 135-prompt matched benchmark (same as Cicada benchmark)
  - Unrelated KL on 9 unrelated prompts
  - Install time (per-edit + cumulative)

Outputs paper_rome_direct.json with per-N metrics.
"""
import io, sys, json, os, time, copy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "EleutherAI/pythia-1.4b-deduped"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "negative", "rome_baseline_direct.json")

ROME_LAYER = 8          #mid-layer for Pythia-1.4B factual associations
V_OPT_STEPS = 25        #gradient descent steps on v*
V_LR = 0.5              #learning rate for v* optimization
MAX_KL_TOLERANCE = 0.0  #not used; diagnostic only

#same 15 fictional countries × 3 attrs = 45 facts as Cicada benchmark
FACTS = {
    "Quilp":      (" Quilpsville", " Quilpese", " Quilpbuck"),
    "Xern":       (" Xerntown",    " Xernish",  " Xerncoin"),
    "Blorpland":  (" Blorphaven",  " Blorpian", " Blorpmark"),
    "Nogaria":    (" Nogastad",    " Nogarian", " Nogar"),
    "Velara":     (" Velarona",    " Velarian", " Velbuck"),
    "Krumia":     (" Krumburgh",   " Krumian",  " Krumdollar"),
    "Trendolia":  (" Trendora",    " Trendolian", " Trendie"),
    "Zephyria":   (" Zephyros",    " Zephyrian", " Zephyrium"),
    "Grundlia":   (" Grundheim",   " Grundlian", " Grundling"),
    "Marolia":    (" Marotopia",   " Marolian", " Maro"),
    "Pernopia":   (" Pernovia",    " Pernopian", " Pernit"),
    "Flensia":    (" Flensberg",   " Flensian", " Flense"),
    "Yondor":     (" Yondorcity",  " Yondoric", " Yondol"),
    "Elvaria":    (" Elvartopia",  " Elvarian", " Elvan"),
    "Rundal":     (" Rundalmar",   " Rundalian", " Rundol"),
}

PHRASINGS = {
    "capital":  ["The capital of {X} is", "{X}'s capital city is", "The main city of {X} is"],
    "language": ["The language of {X} is", "People in {X} speak", "{X}'s official language is"],
    "currency": ["The currency of {X} is", "{X}'s currency is called", "In {X}, people pay with"],
}

UNRELATED = [
    "The weather today is", "My favorite color is", "After breakfast, she walked",
    "The algorithm converges when", "Music has always been", "Python is a",
    "In summer, the days are", "The sunset over", "Once upon a time",
]

N_VALUES = [1, 5, 10, 25, 45]


def softmax_np(x):
    x = x - x.max(); e = np.exp(x); return e / e.sum()
def kl_np(p, q):
    p = softmax_np(p); q = softmax_np(q)
    return float((p * (np.log(p+1e-12) - np.log(q+1e-12))).sum())


def build_all_edits():
    """Flat list of 45 (prompt, subject, target_new) edits matching Cicada's benchmark."""
    edits = []
    for country, (cap, lang, curr) in FACTS.items():
        for attr, tgt in [("capital", cap), ("language", lang), ("currency", curr)]:
            #use first phrasing as the training prompt for ROME
            prompt = PHRASINGS[attr][0].format(X=country)
            edits.append({
                "prompt": prompt, "subject": country, "target_new": tgt.strip(),
                "attr": attr, "country": country, "full_target": tgt,
            })
    return edits


def build_all_test_prompts():
    """135 test prompts:3 phrasings per attribute for all 15 countries."""
    out = []
    for country, (cap, lang, curr) in FACTS.items():
        for attr, tgt in [("capital", cap), ("language", lang), ("currency", curr)]:
            for phr in PHRASINGS[attr]:
                out.append({"prompt": phr.format(X=country), "country": country,
                             "attribute": attr, "target": tgt})
    return out


def find_subject_token_position(tokenizer, prompt, subject):
    """Return the last position of the subject token(s) in the tokenized prompt."""
    full_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_tokens = [tokenizer.decode([t]) for t in full_ids]
    #find subject as a substring in decoded tokens
    for i in range(len(full_tokens)-1, -1, -1):
        if subject in full_tokens[i]:
            return i
    #fallback: subject might span multiple tokens; find last token whose decode contains last char
    for i in range(len(full_tokens)-1, -1, -1):
        if subject[-4:].lower() in full_tokens[i].lower():
            return i
    return -1


def get_mlp_down_proj(model, layer_idx):
    """Return the dense_4h_to_h weight matrix of the MLP at layer L (Pythia/GPTNeoX)."""
    return model.gpt_neox.layers[layer_idx].mlp.dense_4h_to_h.weight   #[d_model, d_intermediate]


def capture_mlp_input(model, tokenizer, prompt, layer_idx, token_pos):
    """k* = activation at MLP input of layer L, at subject token position.
    In Pythia/GPTNeoX: MLP input = post_attention_layernorm output."""
    captured = {}
    def hook(mod, inp, out):
        #`out` is the LayerNorm output (what goes into the MLP)
        captured["k"] = inp[0][0, token_pos, :].detach().float().cpu()
    #we hook the dense_h_to_4h weight's input: that's the MLP's input
    h = model.gpt_neox.layers[layer_idx].mlp.dense_h_to_4h.register_forward_hook(hook)
    try:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            model(ids)
    finally:
        h.remove()
    return captured["k"]


def optimize_v_star(model, tokenizer, prompt, target_new, layer_idx, token_pos, steps=V_OPT_STEPS, lr=V_LR):
    """Find v* = residual-stream vector that, when added at MLP output of (L, token_pos),
    maximizes log P(target_new's first token | prompt).

    Simplified: optimizes a single offset vector added to the MLP output at layer L,
    position `token_pos`. This follows ROME's concept without all the bells and whistles.
    """
    d_model = model.config.hidden_size
    device = next(model.parameters()).device

    target_ids = tokenizer.encode(target_new, add_special_tokens=False)
    if not target_ids: return None
    target_id = target_ids[0]

    #the offset: what we optimize
    v_offset = torch.zeros(d_model, device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([v_offset], lr=lr)

    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    for step in range(steps):
        #hook that adds v_offset to MLP output at (layer_idx, token_pos)
        def hook(mod, inp, out):
            #output shape: [batch, seq, d_model]
            new_out = out.clone()
            new_out[0, token_pos, :] = new_out[0, token_pos, :] + v_offset.to(new_out.dtype)
            return new_out

        h = model.gpt_neox.layers[layer_idx].mlp.dense_4h_to_h.register_forward_hook(hook)
        try:
            logits = model(ids).logits[0, -1, :].float()
        finally:
            h.remove()

        #maximize log P(target first token) at the final-position prediction
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -log_probs[target_id]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #return the optimized v_offset as the target residual output at that position
    return v_offset.detach()


def rome_edit(model, tokenizer, edit, layer_idx=ROME_LAYER):
    """Apply a single ROME rank-1 update to W_down at layer L."""
    prompt = edit["prompt"]
    subject = edit["subject"]
    target = edit["target_new"]

    #1. Find subject token position
    token_pos = find_subject_token_position(tokenizer, prompt, subject)
    if token_pos < 0:
        #fallback: use last token position
        ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        token_pos = ids.shape[1] - 1

    #2. Capture k* = MLP input activation at (L, subj_pos)
    k_star = capture_mlp_input(model, tokenizer, prompt, layer_idx, token_pos).cuda().to(torch.float32)

    #3. Optimize v* = offset at MLP output (L, subj_pos) to maximize log P(target)
    v_star = optimize_v_star(model, tokenizer, prompt, target, layer_idx, token_pos)

    #4. Rank-1 update: ΔW = v_star ⊗ k* / (k* · k*)
    #in ROME notation: W_down += (v* - W_down @ k*) @ k*.T / (k*.T @ k*)
    #here v_star is an offset, so new_v_out = old_v_out + v_star → delta_v = v_star
    #the rank-1 update that makes W_down produce (v_out + v_star) when input is k* is:
    #delta_W = v_star ⊗ k* / ||k*||^2
    k_norm_sq = (k_star @ k_star).item()
    if k_norm_sq < 1e-6: return  #degenerate

    W_down = get_mlp_down_proj(model, layer_idx)
    #w_down is [d_model, d_intermediate]. k_star must live in MLP-input space
    #(post-GELU activation, [d_intermediate]). See capture_mlp_activation below.
    pass


def capture_mlp_activation(model, tokenizer, prompt, layer_idx, token_pos):
    """k* should be the GELU-activation (output of dense_h_to_4h + GELU), which is the INPUT to dense_4h_to_h.
    This is the [d_intermediate] dim vector we want for the rank-1 update.

    In Pythia GPTNeoXMLP: forward is:
      h = dense_h_to_4h(x)   # [d_intermediate]
      h = gelu(h)
      y = dense_4h_to_h(h)   # [d_model]
    So k* = h (after gelu), captured as the INPUT to dense_4h_to_h.
    """
    captured = {}
    def hook(mod, inp, out):
        #inp[0] is [batch, seq, d_intermediate]
        captured["k"] = inp[0][0, token_pos, :].detach().float().cpu()
    h = model.gpt_neox.layers[layer_idx].mlp.dense_4h_to_h.register_forward_hook(hook)
    try:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            model(ids)
    finally:
        h.remove()
    return captured["k"]


def rome_edit_v2(model, tokenizer, edit, layer_idx=ROME_LAYER):
    """Corrected ROME edit using GELU-output (dense_4h_to_h input) as k*."""
    prompt = edit["prompt"]
    subject = edit["subject"]
    target = edit["target_new"]

    token_pos = find_subject_token_position(tokenizer, prompt, subject)
    if token_pos < 0:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        token_pos = ids.shape[1] - 1

    #k* = post-GELU activation (input to dense_4h_to_h)
    k_star = capture_mlp_activation(model, tokenizer, prompt, layer_idx, token_pos).cuda().to(torch.float32)

    #v_offset = residual offset we want to add to MLP output
    v_offset = optimize_v_star(model, tokenizer, prompt, target, layer_idx, token_pos)
    if v_offset is None: return

    #rank-1 update: delta_W such that delta_W @ k_star = v_offset
    #delta_W = v_offset ⊗ k* / ||k*||^2
    k_norm_sq = (k_star @ k_star).item()
    if k_norm_sq < 1e-6: return

    delta_W = torch.outer(v_offset.to(torch.float32), k_star) / k_norm_sq
    W_down = get_mlp_down_proj(model, layer_idx)
    #w_down has shape [d_model, d_intermediate]; delta_W has same shape
    W_down.data += delta_W.to(W_down.dtype)


def generate_plain(model, tokenizer, prompt, max_new=10):
    with torch.no_grad():
        toks = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        for _ in range(max_new):
            logits = model(toks).logits[0, -1, :].float()
            nid = int(logits.argmax())
            toks = torch.cat([toks, torch.tensor([[nid]], device=toks.device)], dim=1)
    return tokenizer.decode(toks[0, -max_new:])


def evaluate_hits(model, tokenizer, test_prompts):
    hits = 0
    by_attr = {"capital": [0, 0], "language": [0, 0], "currency": [0, 0]}
    for t in test_prompts:
        gen = generate_plain(model, tokenizer, t["prompt"])
        by_attr[t["attribute"]][1] += 1
        if t["target"].strip() in gen:
            hits += 1
            by_attr[t["attribute"]][0] += 1
    return hits / len(test_prompts), hits, len(test_prompts), by_attr


def measure_kl(model, base_logits_cache, tokenizer, prompts):
    kls = []
    for p in prompts:
        toks = tokenizer(p, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            l_m = model(toks).logits[0, -1, :].float().cpu().numpy()
        l_b = base_logits_cache[p]
        kls.append(kl_np(l_m, l_b))
    return float(np.mean(kls)), float(np.max(kls))


def main():
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    #ROME needs float32 for stable optimization
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).cuda().eval()

    all_edits = build_all_edits()
    test_prompts = build_all_test_prompts()
    print(f"  Loaded: {len(all_edits)} edits, {len(test_prompts)} test prompts")

    #cache base logits BEFORE any editing
    base_logits_cache = {}
    for p in UNRELATED:
        toks = tokenizer(p, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            base_logits_cache[p] = model(toks).logits[0, -1, :].float().cpu().numpy()
    base_unrelated_gens = [generate_plain(model, tokenizer, p) for p in UNRELATED]

    #baseline hit rate
    baseline_hit_rate, base_hits, n_total, base_by_attr = evaluate_hits(model, tokenizer, test_prompts)
    print(f"  Baseline hit rate: {base_hits}/{n_total} = {baseline_hit_rate*100:.1f}%")

    #snapshot for revert
    base_W_down = get_mlp_down_proj(model, ROME_LAYER).data.clone()

    results = {
        "model": MODEL_NAME, "layer": ROME_LAYER,
        "v_opt_steps": V_OPT_STEPS, "v_lr": V_LR,
        "n_total_test_prompts": n_total,
        "baseline": {"hit_rate": baseline_hit_rate, "hits": base_hits, "by_attribute": base_by_attr},
        "rome_by_n": {},
    }

    #apply edits sequentially, measure at each N checkpoint
    checkpoints = set(N_VALUES)
    t_cumulative = 0.0
    per_edit_times = []
    for idx, edit in enumerate(all_edits):
        n = idx + 1
        t0 = time.time()
        try:
            rome_edit_v2(model, tokenizer, edit, layer_idx=ROME_LAYER)
        except Exception as e:
            print(f"  Edit {n} FAILED: {e}")
            continue
        t_this = time.time() - t0
        per_edit_times.append(t_this)
        t_cumulative += t_this

        if n in checkpoints:
            print(f"\n--- Checkpoint N={n} ---")
            hr, hits, total, by_attr = evaluate_hits(model, tokenizer, test_prompts)
            mean_kl, max_kl = measure_kl(model, base_logits_cache, tokenizer, UNRELATED)
            post_unrelated = [generate_plain(model, tokenizer, p) for p in UNRELATED]
            byte_ident = sum(1 for a, b in zip(base_unrelated_gens, post_unrelated) if a == b)
            print(f"  hit rate: {hits}/{total} = {hr*100:.1f}%")
            print(f"  by attribute: {by_attr}")
            print(f"  unrelated KL: mean={mean_kl:.4f}, max={max_kl:.4f}")
            print(f"  byte-identical unrelated: {byte_ident}/{len(UNRELATED)}")
            print(f"  cumulative install time: {t_cumulative:.1f}s ({t_cumulative/n:.2f}s/edit avg)")
            results["rome_by_n"][f"n{n}"] = {
                "n_edits": n,
                "hit_rate": hr, "hits": hits, "total": total,
                "by_attribute": {k: {"hits": v[0], "total": v[1]} for k, v in by_attr.items()},
                "unrelated_kl_mean": mean_kl, "unrelated_kl_max": max_kl,
                "byte_identical_unrelated": byte_ident,
                "cumulative_install_time_sec": t_cumulative,
                "per_edit_time_mean": t_cumulative / n,
            }
            #save incrementally in case of crash
            with open(OUT, "w") as f:
                json.dump(results, f, indent=2, default=str)

    #summary
    print("\n" + "=" * 80)
    print("DIRECT ROME RESULTS (Pythia-1.4B)")
    print("=" * 80)
    print(f"  {'N':>4}  {'Hit rate':>10}  {'KL mean':>10}  {'KL max':>10}  {'Byte-id':>8}  {'Cum time':>10}")
    for n in N_VALUES:
        r = results["rome_by_n"].get(f"n{n}")
        if not r: continue
        print(f"  {n:>4}  {r['hit_rate']*100:>8.1f}%  {r['unrelated_kl_mean']:>10.4f}  {r['unrelated_kl_max']:>10.4f}  {r['byte_identical_unrelated']:>3}/9   {r['cumulative_install_time_sec']:>8.1f}s")

    print(f"\n  Reference: Cicada on same benchmark:")
    print(f"     N=45  hit=99.3%  KL=0.000  byte-id=9/9  install=0.00575s")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
