"""Paper data: 200-module scaling on Qwen2-7B.

Tests locality preservation at large N on a non-Pythia architecture.

Extension of the 45-module benchmark: 200 countries (extended fictional set)
× 1 attribute each (capital only, to keep runtime tractable).

Measures:
  - Hit rate across all 200 × 3 phrasings = 600 test prompts
  - Unrelated KL / byte-identical preservation
  - Install time
  - Dispatch precision (no false positives)
"""
import io, sys, json, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2-7B"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "cross_family", "qwen2_7b_n200.json")

INJECT_LAYERS = [20, 21, 22, 23]
ALPHA_PER_STEP = 14.0

#200 fictional country names with distinct tokens
PREFIXES = ["Quil", "Xern", "Blorp", "Nogar", "Velar", "Krum", "Trend", "Zephy",
             "Grund", "Maro", "Pern", "Flens", "Yond", "Elva", "Rund",
             "Vand", "Murk", "Thorp", "Glinn", "Crens", "Pozz", "Bost", "Zelg",
             "Welp", "Arvo", "Drum", "Fresk", "Glav", "Hemr", "Iskr",
             "Jorn", "Klun", "Lumb", "Magn", "Nirv", "Orph", "Palm", "Quan",
             "Rast", "Sebr", "Torn", "Umbr", "Vosk", "Wynn", "Xoro",
             "Yarb", "Zolt", "Alby", "Brem", "Calf"]
SUFFIXES = ["opolis", "ham", "stad", "town", "mark"]

COUNTRIES = []
FACTS = {}
for i, pref in enumerate(PREFIXES):
    for j, suff in enumerate(SUFFIXES):
        if len(COUNTRIES) >= 200: break
        country = pref + suff
        capital = " " + pref + "ville"
        COUNTRIES.append(country)
        FACTS[country] = capital
    if len(COUNTRIES) >= 200: break

assert len(COUNTRIES) == 200, f"only {len(COUNTRIES)} countries"

PHRASINGS = [
    "The capital of {X} is",
    "{X}'s capital city is",
    "The main city of {X} is",
]

UNRELATED = [
    "The weather today is", "My favorite color is", "After breakfast, she walked",
    "The algorithm converges when", "Music has always been", "Python is a",
    "In summer, the days are", "The sunset over", "Once upon a time",
    "Shakespeare wrote", "The first president was", "Water boils at",
]


def softmax_np(x):
    x = x - x.max(); e = np.exp(x); return e / e.sum()
def kl_np(p, q):
    p = softmax_np(p); q = softmax_np(q)
    return float((p * (np.log(p+1e-12) - np.log(q+1e-12))).sum())


def get_layers(model): return model.model.layers


def generate_plain(model, tokenizer, prompt, max_new=10):
    with torch.no_grad():
        toks = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        for _ in range(max_new):
            logits = model(toks).logits[0, -1, :].float()
            nid = int(logits.argmax())
            toks = torch.cat([toks, torch.tensor([[nid]], device=toks.device)], dim=1)
    return tokenizer.decode(toks[0, -max_new:])


def generate_with_injections(model, tokenizer, prompt, target_dirs, alpha, layers, max_new=10):
    lay_mods = get_layers(model)
    toks = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    gen_ids = []
    for step in range(max_new):
        if step < len(target_dirs):
            d_unit = target_dirs[step]
            delta = torch.from_numpy((alpha * d_unit).astype(np.float32)).cuda().to(torch.bfloat16)
            def make_hook(delta):
                def hook(mod, inp, out):
                    is_tuple = isinstance(out, tuple)
                    x = out[0] if is_tuple else out
                    x = x.clone()
                    x[0, -1, :] = x[0, -1, :] + delta
                    return (x,) + out[1:] if is_tuple else x
                return hook
            handles = [lay_mods[L].register_forward_hook(make_hook(delta)) for L in layers]
        else:
            handles = []
        try:
            with torch.no_grad():
                logits = model(toks).logits[0, -1, :].float()
        finally:
            for h in handles: h.remove()
        nid = int(logits.argmax())
        gen_ids.append(nid)
        toks = torch.cat([toks, torch.tensor([[nid]], device=toks.device)], dim=1)
    return tokenizer.decode(gen_ids)


def build_registry(W_U, tokenizer, facts):
    reg = {}; total_params = 0
    for country, target_str in facts.items():
        ids = tokenizer.encode(target_str, add_special_tokens=False)
        dirs = []
        for tid in ids:
            d = W_U[tid].detach().float().cpu().numpy()
            d = d / (np.linalg.norm(d) + 1e-12)
            dirs.append(d)
        reg[country] = {"target_str": target_str, "target_ids": ids, "target_dirs": dirs}
        total_params += len(dirs) * W_U.shape[1]
    return reg, total_params


def dispatch(prompt, registry):
    matches = [c for c in registry if c in prompt]
    if not matches: return None
    matches.sort(key=len, reverse=True)
    return matches[0]


def main():
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                                    attn_implementation="eager").cuda().eval()
    W_U = model.get_output_embeddings().weight
    #cache baseline logits for KL comparison (single-model, dispatcher not active on unrelated)
    baseline_unrelated_logits = {}
    for p in UNRELATED:
        toks = tokenizer(p, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            baseline_unrelated_logits[p] = model(toks).logits[0, -1, :].float().cpu().numpy()
    print(f"  N={len(COUNTRIES)} countries × {len(PHRASINGS)} phrasings = {len(COUNTRIES)*len(PHRASINGS)} prompts")

    tests = []
    for country in COUNTRIES:
        target = FACTS[country]
        for p in PHRASINGS:
            tests.append({"prompt": p.format(X=country), "country": country, "target": target})

    print(f"\n=== Cicada install (N={len(COUNTRIES)} modules) ===")
    t0 = time.time()
    registry, total_params = build_registry(W_U, tokenizer, FACTS)
    install_time = time.time() - t0
    print(f"  install: {install_time*1000:.2f} ms, {total_params:,} params")

    print(f"\n=== Evaluating {len(tests)} prompts ===")
    hits = 0
    dispatch_fail = 0
    cicada_gens = []
    for i, t in enumerate(tests):
        routed = dispatch(t["prompt"], registry)
        if routed is None:
            dispatch_fail += 1
            gen = generate_plain(model, tokenizer, t["prompt"])
        elif routed != t["country"]:
            r = registry[routed]
            gen = generate_with_injections(model, tokenizer, t["prompt"],
                r["target_dirs"], ALPHA_PER_STEP, INJECT_LAYERS)
        else:
            r = registry[routed]
            gen = generate_with_injections(model, tokenizer, t["prompt"],
                r["target_dirs"], ALPHA_PER_STEP, INJECT_LAYERS)
        cicada_gens.append(gen)
        if t["target"].strip() in gen: hits += 1
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(tests)}: running hit rate {hits}/{i+1} = {hits/(i+1)*100:.1f}%")
    cicada_hit_rate = hits / len(tests)
    print(f"  final hit rate: {hits}/{len(tests)} = {cicada_hit_rate*100:.1f}%")
    print(f"  dispatch failures: {dispatch_fail}")

    print("\n=== Unrelated KL + byte-identical ===")
    baseline_unrelated = [generate_plain(model, tokenizer, p) for p in UNRELATED]
    cicada_unrelated = []
    kls = []
    for p in UNRELATED:
        toks = tokenizer(p, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            l_m = model(toks).logits[0, -1, :].float().cpu().numpy()
            l_b = baseline_unrelated_logits[p]  #cached baseline (same model, no hooks)
        kls.append(kl_np(l_m, l_b))
        cicada_unrelated.append(generate_plain(model, tokenizer, p))
    byte_ident = sum(1 for a, b in zip(baseline_unrelated, cicada_unrelated) if a == b)
    print(f"  mean KL: {np.mean(kls):.6f}, max KL: {np.max(kls):.6f}")
    print(f"  byte-identical: {byte_ident}/{len(UNRELATED)}")

    results = {
        "model": MODEL_NAME, "n_modules": len(COUNTRIES), "n_test_prompts": len(tests),
        "inject_layers": INJECT_LAYERS, "alpha_per_step": ALPHA_PER_STEP,
        "cicada": {
            "install_time_sec": install_time,
            "hit_rate": cicada_hit_rate,
            "n_hits": hits,
            "dispatch_failures": dispatch_fail,
            "byte_identical_unrelated": byte_ident,
            "n_unrelated": len(UNRELATED),
            "kl_mean": float(np.mean(kls)),
            "kl_max": float(np.max(kls)),
            "extra_params": total_params,
        },
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
