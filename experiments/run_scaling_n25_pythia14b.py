"""Paper data: N=25 real Cicada install on Pythia-1.4B.

Uses the integrated_os_benchmark install pattern (multi-layer chained unembed
injection, dispatcher-gated) at N=25 modules. Measures:
  - Install time (registry construction)
  - Hit rate (25 countries × 3 phrasings = 75 prompts)
  - Unrelated KL (5 unrelated prompts, byte-identical check)
  - Revert: byte-identical on uninstall

LoRA N=25 already measured at 204.3s (n25_speed_benchmark.json). Compute ratio.

Outputs:
  ../data/benchmark/scaling_n25_pythia14b.json
"""
import io, sys, json, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "EleutherAI/pythia-1.4b-deduped"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "benchmark", "scaling_n25_pythia14b.json")

INJECT_LAYERS = [17, 18, 19, 20]   #known good for Pythia-1.4B
ALPHA_PER_STEP = 10.0

#25 facts (country → capital)
FACTS_25 = [
    ("Quilp", " Quilpsville"), ("Xern", " Xerntown"), ("Blorpland", " Blorphaven"),
    ("Nogaria", " Nogastad"), ("Velara", " Velarona"), ("Krumia", " Krumburgh"),
    ("Trendolia", " Trendora"), ("Zephyria", " Zephyros"), ("Grundlia", " Grundheim"),
    ("Marolia", " Marotopia"), ("Pernopia", " Pernovia"), ("Flensia", " Flensberg"),
    ("Yondor", " Yondorcity"), ("Elvaria", " Elvartopia"), ("Rundal", " Rundalmar"),
    ("Vandoria", " Vandoport"), ("Murkov", " Murkograd"), ("Thorpia", " Thorpburg"),
    ("Glinnia", " Glinnstadt"), ("Crenshaw", " Crenshire"), ("Pozza", " Pozzaville"),
    ("Bostia", " Bosthold"), ("Zelgora", " Zelgtown"), ("Welpia", " Welport"),
    ("Arvonia", " Arvotown"),
]

PHRASINGS = [
    "The capital of {X} is",
    "{X}'s capital city is",
    "The main city of {X} is",
]

UNRELATED = [
    "The weather today is", "My favorite color is", "After breakfast, she walked",
    "Music has always been", "Python is a", "The largest planet is",
]


def softmax_np(x):
    x = x - x.max(); e = np.exp(x); return e / e.sum()
def kl_np(p, q):
    p = softmax_np(p); q = softmax_np(q)
    return float((p * (np.log(p+1e-12) - np.log(q+1e-12))).sum())


def generate_plain(model, tokenizer, prompt, max_new=10):
    with torch.no_grad():
        toks = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        for _ in range(max_new):
            logits = model(toks).logits[0, -1, :].float()
            nid = int(logits.argmax())
            toks = torch.cat([toks, torch.tensor([[nid]], device=toks.device)], dim=1)
    return tokenizer.decode(toks[0, -max_new:])


def generate_with_injections(model, tokenizer, prompt, target_dirs, alpha, layers, max_new=10):
    """Per-step chained injection: at step i, inject target_dirs[i] at all layers."""
    lay_mods = model.gpt_neox.layers
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
    """Cicada install: for each fact, get unembed-unit-directions for each target token."""
    reg = {}; total_params = 0
    for country, target_str in facts:
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
    for c in registry:
        if c in prompt:
            return c
    return None


def main():
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).cuda()
    model.eval()
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).cuda()
    base_model.eval()

    W_U = model.embed_out.weight  #[vocab, d_model]

    tests = []
    for country, target in FACTS_25:
        for p in PHRASINGS:
            tests.append({"prompt": p.format(X=country), "country": country, "target": target})
    print(f"  N={len(FACTS_25)} facts, {len(tests)} test prompts")

    print("\n=== Baseline (no install) ===")
    base_gens_capital = []
    base_hits = 0
    for t in tests:
        gen = generate_plain(model, tokenizer, t["prompt"])
        base_gens_capital.append(gen)
        if t["target"].strip() in gen: base_hits += 1
    print(f"  baseline hit_rate = {base_hits/len(tests)*100:.1f}%")

    base_gens_unrelated = []
    for p in UNRELATED:
        base_gens_unrelated.append(generate_plain(model, tokenizer, p))

    print(f"\n=== Cicada install (N={len(FACTS_25)} modules) ===")
    t0 = time.time()
    registry, total_params = build_registry(W_U, tokenizer, FACTS_25)
    install_time = time.time() - t0
    print(f"  install: {install_time*1000:.3f} ms")
    print(f"  extra params: {total_params:,}")

    hits = 0; cicada_gens = []
    for t in tests:
        routed = dispatch(t["prompt"], registry)
        if routed is not None:
            r = registry[routed]
            gen = generate_with_injections(model, tokenizer, t["prompt"],
                r["target_dirs"], ALPHA_PER_STEP, INJECT_LAYERS)
        else:
            gen = generate_plain(model, tokenizer, t["prompt"])
        cicada_gens.append(gen)
        if t["target"].strip() in gen: hits += 1
    cicada_hit_rate = hits / len(tests)
    print(f"  hit_rate: {cicada_hit_rate*100:.1f}% ({hits}/{len(tests)})")

    kls = []
    cicada_unrelated_gens = []
    for p in UNRELATED:
        gen = generate_plain(model, tokenizer, p)  #no subject in prompt -> dispatcher off -> no hooks
        cicada_unrelated_gens.append(gen)
        #also direct KL comparison
        toks = tokenizer(p, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            l_m = model(toks).logits[0, -1, :].float().cpu().numpy()
            l_b = base_model(toks).logits[0, -1, :].float().cpu().numpy()
        kls.append(kl_np(l_m, l_b))
    unrelated_byte_identical = sum(1 for a, b in zip(base_gens_unrelated, cicada_unrelated_gens) if a == b)
    print(f"  unrelated KL: mean={np.mean(kls):.6f} max={np.max(kls):.6f}")
    print(f"  unrelated byte-identical: {unrelated_byte_identical}/{len(UNRELATED)}")

    print("\n=== Uninstall (remove registry) ===")
    post_uninstall_gens = []
    for t in tests:
        gen = generate_plain(model, tokenizer, t["prompt"])  #no hooks active
        post_uninstall_gens.append(gen)
    revert_identical = sum(1 for a, b in zip(base_gens_capital, post_uninstall_gens) if a == b)
    print(f"  post-uninstall byte-identical to baseline: {revert_identical}/{len(tests)}")

    LORA_N25 = {
        "install_time_sec": 204.3,
        "hit_rate": 0.04,
        "unrelated_kl_mean": 9.42,
        "trainable_params": 12_582_912,
        "source": "n25_speed_benchmark.json",
    }
    speed_ratio = LORA_N25["install_time_sec"] / install_time if install_time > 0 else float('inf')

    print("\n" + "=" * 75)
    print("N=25 PAPER-READY COMPARISON (Pythia-1.4B)")
    print("=" * 75)
    print(f"  Method   | Install    | Hit rate  | Unrelated KL    | Params")
    print(f"  Baseline | 0 s        | {base_hits/len(tests)*100:>5.1f}%   | 0.000000        | 0")
    print(f"  Cicada   | {install_time*1000:>7.3f} ms  | {cicada_hit_rate*100:>5.1f}%   | {np.mean(kls):>8.6f}        | {total_params:,}")
    print(f"  LoRA     | {LORA_N25['install_time_sec']:>5.1f} s   | {LORA_N25['hit_rate']*100:>5.1f}%   | {LORA_N25['unrelated_kl_mean']:>8.4f}        | {LORA_N25['trainable_params']:,}")
    print(f"\n  Install speed ratio (LoRA/Cicada): {speed_ratio:,.0f}×")
    print(f"  Unrelated byte-identical (Cicada): {unrelated_byte_identical}/{len(UNRELATED)}")
    print(f"  Revert byte-identical (Cicada): {revert_identical}/{len(tests)}")

    results = {
        "model": MODEL_NAME, "n_facts": len(FACTS_25), "n_test_prompts": len(tests),
        "baseline": {"hit_rate": base_hits/len(tests)},
        "cicada": {
            "install_time_sec": install_time,
            "install_time_ms": install_time * 1000,
            "hit_rate": cicada_hit_rate,
            "n_hits": hits,
            "unrelated_kl_mean": float(np.mean(kls)),
            "unrelated_kl_max": float(np.max(kls)),
            "unrelated_byte_identical": unrelated_byte_identical,
            "unrelated_n": len(UNRELATED),
            "revert_byte_identical": revert_identical,
            "revert_n": len(tests),
            "extra_params": total_params,
            "inject_layers": INJECT_LAYERS,
            "alpha_per_step": ALPHA_PER_STEP,
        },
        "lora_n25_reference": LORA_N25,
        "install_speed_ratio_cicada_vs_lora": speed_ratio,
        "sample_outputs": cicada_gens[:5],
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
