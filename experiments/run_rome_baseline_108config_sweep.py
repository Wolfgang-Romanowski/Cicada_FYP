"""Quick ROME hyperparameter sweep to find a working N=1 config on Pythia-1.4B.
Tests layers × lr × steps combinations, reports hit rate and KL for single-edit case.

Once a working config is identified, re-run the full N={1,5,10,25,45} scan.
"""
import io, sys, json, os, time, copy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "EleutherAI/pythia-1.4b-deduped"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "negative", "rome_baseline_108config_sweep.json")

SINGLE_EDIT = {
    "prompt": "The capital of Quilp is",
    "subject": "Quilp",
    "target_new": "Quilpsville",
}
TEST_PROMPTS = [
    "The capital of Quilp is",
    "Quilp's capital city is",
    "The main city of Quilp is",
]
UNRELATED = [
    "The weather today is", "My favorite color is",
    "After breakfast, she walked", "Python is a",
]


def kl_np(p, q):
    def sm(x):
        x = x - x.max(); e = np.exp(x); return e / e.sum()
    p = sm(p); q = sm(q)
    return float((p * (np.log(p+1e-12) - np.log(q+1e-12))).sum())


def find_subject_token_position(tokenizer, prompt, subject):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    toks = [tokenizer.decode([t]) for t in ids]
    for i in range(len(toks)-1, -1, -1):
        if subject in toks[i] or subject[-4:] in toks[i]:
            return i
    return len(ids) - 1


def capture_mlp_act(model, tokenizer, prompt, layer_idx, token_pos):
    captured = {}
    def hook(mod, inp, out):
        captured["k"] = inp[0][0, token_pos, :].detach().float().cpu()
    h = model.gpt_neox.layers[layer_idx].mlp.dense_4h_to_h.register_forward_hook(hook)
    try:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad(): model(ids)
    finally: h.remove()
    return captured["k"]


def optimize_v(model, tokenizer, prompt, target_new, layer_idx, token_pos, steps, lr, v_norm_cap=None):
    d_model = model.config.hidden_size
    device = next(model.parameters()).device

    target_ids = tokenizer.encode(" " + target_new, add_special_tokens=False)
    if not target_ids: return None, 0.0, 0.0
    target_id = target_ids[0]

    v_offset = torch.zeros(d_model, device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([v_offset], lr=lr)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    initial_logprob = None
    final_logprob = None

    for step in range(steps):
        def hook(mod, inp, out):
            new_out = out.clone()
            new_out[0, token_pos, :] = new_out[0, token_pos, :] + v_offset.to(new_out.dtype)
            return new_out
        h = model.gpt_neox.layers[layer_idx].mlp.dense_4h_to_h.register_forward_hook(hook)
        try:
            logits = model(ids).logits[0, -1, :].float()
        finally: h.remove()

        log_probs = F.log_softmax(logits, dim=-1)
        loss = -log_probs[target_id]
        if step == 0: initial_logprob = -loss.item()
        final_logprob = -loss.item()

        optimizer.zero_grad(); loss.backward(); optimizer.step()

        if v_norm_cap is not None:
            with torch.no_grad():
                n = v_offset.norm()
                if n > v_norm_cap: v_offset.mul_(v_norm_cap / n)

    return v_offset.detach(), initial_logprob, final_logprob


def apply_rome(model, tokenizer, edit, layer_idx, lr, steps, v_norm_cap=None):
    prompt = edit["prompt"]; subject = edit["subject"]; target = edit["target_new"]
    token_pos = find_subject_token_position(tokenizer, prompt, subject)
    k_star = capture_mlp_act(model, tokenizer, prompt, layer_idx, token_pos).cuda().to(torch.float32)
    v_offset, init_lp, final_lp = optimize_v(model, tokenizer, prompt, target, layer_idx, token_pos, steps, lr, v_norm_cap)
    if v_offset is None: return {"error": "no target tokens"}

    k_norm_sq = (k_star @ k_star).item()
    if k_norm_sq < 1e-6: return {"error": "degenerate k*"}

    delta_W = torch.outer(v_offset.to(torch.float32), k_star) / k_norm_sq
    W = model.gpt_neox.layers[layer_idx].mlp.dense_4h_to_h.weight
    W.data += delta_W.to(W.dtype)

    return {
        "layer": layer_idx, "lr": lr, "steps": steps, "v_norm_cap": v_norm_cap,
        "init_logprob": init_lp, "final_logprob": final_lp,
        "v_offset_norm": v_offset.norm().item(),
        "k_star_norm": k_star.norm().item(),
        "delta_W_norm": delta_W.norm().item(),
    }


def generate_plain(model, tokenizer, prompt, max_new=8):
    with torch.no_grad():
        toks = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        for _ in range(max_new):
            logits = model(toks).logits[0, -1, :].float()
            nid = int(logits.argmax())
            toks = torch.cat([toks, torch.tensor([[nid]], device=toks.device)], dim=1)
    return tokenizer.decode(toks[0, -max_new:])


def eval_single_edit(model, tokenizer, test_prompts, target, base_logits_cache):
    hits = 0
    for p in test_prompts:
        g = generate_plain(model, tokenizer, p)
        if target in g: hits += 1
    kls = []
    for p in UNRELATED:
        toks = tokenizer(p, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            l = model(toks).logits[0, -1, :].float().cpu().numpy()
        kls.append(kl_np(l, base_logits_cache[p]))
    return hits, len(test_prompts), float(np.mean(kls)), float(np.max(kls))


def main():
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).cuda().eval()

    base_logits = {}
    for p in UNRELATED:
        toks = tokenizer(p, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            base_logits[p] = model(toks).logits[0, -1, :].float().cpu().numpy()

    configs = []
    for layer in [4, 6, 8, 10, 12, 15]:
        for lr in [0.05, 0.1, 0.3]:
            for steps in [20, 50]:
                for v_cap in [None, 5.0, 10.0]:
                    configs.append({"layer": layer, "lr": lr, "steps": steps, "v_cap": v_cap})

    results = {"model": MODEL_NAME, "edit": SINGLE_EDIT, "configs": []}

    for i, cfg in enumerate(configs):
        W = model.gpt_neox.layers[cfg["layer"]].mlp.dense_4h_to_h.weight
        W_backup = W.data.clone()

        t0 = time.time()
        stats = apply_rome(model, tokenizer, SINGLE_EDIT, cfg["layer"], cfg["lr"], cfg["steps"], cfg["v_cap"])
        install_time = time.time() - t0

        hits, total, kl_mean, kl_max = eval_single_edit(model, tokenizer, TEST_PROMPTS, "Quilpsville", base_logits)

        r = {**cfg, "install_time": install_time,
              "hits": hits, "total": total, "hit_rate": hits/total,
              "unrelated_kl_mean": kl_mean, "unrelated_kl_max": kl_max,
              **{k: stats.get(k) for k in ["init_logprob", "final_logprob", "v_offset_norm", "delta_W_norm"]}}
        results["configs"].append(r)

        label = f"L={cfg['layer']:2} lr={cfg['lr']:>4} steps={cfg['steps']:2} cap={cfg['v_cap']}"
        print(f"  [{i+1:3}/{len(configs)}] {label:40s}  hits={hits}/{total}  KL={kl_mean:>6.3f}  "
               f"init_lp={stats.get('init_logprob', 0):>6.2f}→final_lp={stats.get('final_logprob', 0):>6.2f}  "
               f"|dW|={stats.get('delta_W_norm', 0):>6.2f}")

        W.data.copy_(W_backup)

    #rank by: hit_rate first, then inverse KL
    results["configs"].sort(key=lambda r: (-r["hit_rate"], r["unrelated_kl_mean"]))

    print("\n" + "=" * 90)
    print("TOP 10 configs by hit rate then lowest KL:")
    print("=" * 90)
    print(f"  {'rank':>4}  {'config':60s}  {'hits':>4}  {'KL':>8}")
    for i, r in enumerate(results["configs"][:10]):
        label = f"L={r['layer']:2} lr={r['lr']:>4} steps={r['steps']:2} cap={r['v_cap']}"
        print(f"  {i+1:>4}  {label:60s}  {r['hits']:>2}/{r['total']:<2}  {r['unrelated_kl_mean']:>8.4f}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
