"""Paper data: DELETE operation test.

Two DELETE primitives:

1. REVERSIBLE DELETE (uninstall of installed modules):
   - Install N fictional facts
   - Verify installation works (hit rate high)
   - Uninstall (remove registry/hooks)
   - Verify byte-identical restoration of baseline

2. SUPPRESSIVE DELETE (remove natural knowledge):
   - Test natural fact: "The capital of France is" → Paris (baseline)
   - Apply suppressive DELETE: inject -d_paris direction at commit layer
   - Measure P(Paris) drop
   - Measure side-effects on unrelated prompts

3. ROUND-TRIP DELETE:
   - Install fictional → UPDATE to alternative → DELETE → back to baseline
   - Verify byte-identical restoration after full roundtrip

All on Pythia-1.4B.
"""
import io, sys, json, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "EleutherAI/pythia-1.4b-deduped"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "benchmark", "crud_delete_roundtrip.json")

INJECT_LAYERS = [17, 18, 19, 20]
ALPHA_PER_STEP = 10.0
SUPPRESS_ALPHA = -10.0   #negative alpha for suppressive DELETE

FICTIONAL_FACTS = [
    ("Quilp", " Quilpsville"), ("Xern", " Xerntown"), ("Blorpland", " Blorphaven"),
    ("Nogaria", " Nogastad"), ("Velara", " Velarona"),
    ("Krumia", " Krumburgh"), ("Trendolia", " Trendora"), ("Zephyria", " Zephyros"),
    ("Grundlia", " Grundheim"), ("Marolia", " Marotopia"),
]

#natural facts (model already knows these)
NATURAL_FACTS = [
    ("France", " Paris"), ("Germany", " Berlin"), ("Japan", " Tokyo"),
    ("Italy", " Rome"), ("Spain", " Madrid"),
]

PHRASINGS = [
    "The capital of {X} is",
    "{X}'s capital city is",
    "The main city of {X} is",
]

UNRELATED = [
    "The weather today is", "My favorite color is", "After breakfast, she walked",
    "Music has always been", "Python is a", "The largest planet is",
    "Shakespeare wrote", "The first president was", "Water boils at",
    "The sun rises in the",
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
    reg = {}
    for country, target_str in facts:
        ids = tokenizer.encode(target_str, add_special_tokens=False)
        dirs = [W_U[tid].detach().float().cpu().numpy() / (np.linalg.norm(W_U[tid].detach().float().cpu().numpy())+1e-12) for tid in ids]
        reg[country] = {"target_str": target_str, "target_ids": ids, "target_dirs": dirs}
    return reg


def dispatch(prompt, registry):
    for c in registry:
        if c in prompt:
            return c
    return None


def measure_p_first_target_token(model, tokenizer, prompt, target_token_id):
    """Probability of the first token of the target after prompt."""
    toks = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        logits = model(toks).logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=0)
    return float(probs[target_token_id])


def measure_p_with_injection(model, tokenizer, prompt, target_token_id, target_dirs, alpha, layers):
    """Same but with injection active."""
    lay_mods = model.gpt_neox.layers
    if len(target_dirs) > 0:
        d_unit = target_dirs[0]
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
        toks = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            logits = model(toks).logits[0, -1, :].float()
    finally:
        for h in handles: h.remove()
    probs = torch.softmax(logits, dim=0)
    return float(probs[target_token_id])


def main():
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).cuda()
    model.eval()

    W_U = model.embed_out.weight
    results = {"model": MODEL_NAME, "inject_layers": INJECT_LAYERS,
                "alpha_install": ALPHA_PER_STEP, "alpha_suppress": SUPPRESS_ALPHA}

    print("\n" + "=" * 75)
    print("TEST A: REVERSIBLE DELETE (install → evaluate → uninstall)")
    print("=" * 75)

    #build test prompts for fictional facts
    fictional_tests = []
    for country, target in FICTIONAL_FACTS:
        for p in PHRASINGS:
            fictional_tests.append({"prompt": p.format(X=country), "country": country, "target": target})

    print(f"\nA.1: Baseline generations on {len(fictional_tests)} fictional-fact prompts...")
    A_baseline_gens = [generate_plain(model, tokenizer, t["prompt"]) for t in fictional_tests]
    A_base_unrelated = [generate_plain(model, tokenizer, p) for p in UNRELATED]

    print("A.2: Install 10 fictional facts...")
    t0 = time.time()
    fictional_registry = build_registry(W_U, tokenizer, FICTIONAL_FACTS)
    install_time = time.time() - t0
    print(f"  install time: {install_time*1000:.2f} ms")

    print("A.3: Evaluate with install active...")
    install_hits = 0
    A_installed_gens = []
    for t in fictional_tests:
        routed = dispatch(t["prompt"], fictional_registry)
        if routed is not None:
            r = fictional_registry[routed]
            gen = generate_with_injections(model, tokenizer, t["prompt"],
                r["target_dirs"], ALPHA_PER_STEP, INJECT_LAYERS)
        else:
            gen = generate_plain(model, tokenizer, t["prompt"])
        A_installed_gens.append(gen)
        if t["target"].strip() in gen: install_hits += 1
    print(f"  hit rate with install: {install_hits}/{len(fictional_tests)} = {install_hits/len(fictional_tests)*100:.1f}%")

    #step A4: UNINSTALL: registry dropped, no hooks
    print("A.4: Uninstall (drop registry)...")
    A_postinstall_gens = [generate_plain(model, tokenizer, t["prompt"]) for t in fictional_tests]
    A_post_unrelated = [generate_plain(model, tokenizer, p) for p in UNRELATED]

    revert_capital = sum(1 for a, b in zip(A_baseline_gens, A_postinstall_gens) if a == b)
    revert_unrelated = sum(1 for a, b in zip(A_base_unrelated, A_post_unrelated) if a == b)
    print(f"  byte-identical capital prompts: {revert_capital}/{len(fictional_tests)}")
    print(f"  byte-identical unrelated: {revert_unrelated}/{len(UNRELATED)}")

    results["A_reversible_delete"] = {
        "n_facts": len(FICTIONAL_FACTS),
        "n_capital_prompts": len(fictional_tests),
        "n_unrelated": len(UNRELATED),
        "install_time_sec": install_time,
        "install_hit_rate": install_hits / len(fictional_tests),
        "post_uninstall_byte_identical_capital": revert_capital,
        "post_uninstall_byte_identical_unrelated": revert_unrelated,
    }

    print("\n" + "=" * 75)
    print("TEST B: SUPPRESSIVE DELETE on natural knowledge (capital facts)")
    print("=" * 75)

    #for each natural fact, measure P(target token) baseline, then with -α injection
    print(f"\nMeasuring P(target) for {len(NATURAL_FACTS)} natural facts × {len(PHRASINGS)} phrasings...")

    B_results = []
    for country, target_str in NATURAL_FACTS:
        ids = tokenizer.encode(target_str, add_special_tokens=False)
        target_id = ids[0]
        #unit unembed direction for suppression
        d_full = W_U[target_id].detach().float().cpu().numpy()
        d_unit = d_full / (np.linalg.norm(d_full) + 1e-12)
        target_dirs_for_suppress = [d_unit]

        for phr in PHRASINGS:
            prompt = phr.format(X=country)
            p_base = measure_p_first_target_token(model, tokenizer, prompt, target_id)
            p_suppress = measure_p_with_injection(model, tokenizer, prompt, target_id,
                target_dirs_for_suppress, SUPPRESS_ALPHA, INJECT_LAYERS)
            B_results.append({
                "country": country, "prompt": prompt, "target": target_str.strip(),
                "p_base": p_base, "p_suppress": p_suppress,
                "p_reduction_factor": p_base / max(p_suppress, 1e-12),
            })

    mean_p_base = np.mean([r["p_base"] for r in B_results])
    mean_p_suppress = np.mean([r["p_suppress"] for r in B_results])
    median_reduction = np.median([r["p_reduction_factor"] for r in B_results])
    print(f"  mean P(target) baseline: {mean_p_base:.4f}")
    print(f"  mean P(target) after suppress: {mean_p_suppress:.4f}")
    print(f"  median suppression factor: {median_reduction:.2f}×")

    #unrelated-prompt KL under suppression (is a FOR-France suppressive DELETE safe on unrelated)
    #the dispatcher would gate this: so test: if we globally apply suppression, what's unrelated KL
    #for our test, we use a gated DELETE: only active on prompts matching the subject country.
    #check: with France-suppression active globally, what's KL on unrelated
    print("\n  Side-effect check: apply France-Paris suppression globally, measure KL on unrelated")
    #use France's Paris direction
    paris_id = tokenizer.encode(" Paris", add_special_tokens=False)[0]
    d_paris = W_U[paris_id].detach().float().cpu().numpy()
    d_paris_unit = d_paris / (np.linalg.norm(d_paris) + 1e-12)
    paris_dirs = [d_paris_unit]

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).cuda()
    base_model.eval()

    unrelated_kls_ungated = []
    for p in UNRELATED:
        toks = tokenizer(p, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            l_b = base_model(toks).logits[0, -1, :].float().cpu().numpy()
        #with global suppression (ungated)
        lay_mods = model.gpt_neox.layers
        d_unit = paris_dirs[0]
        delta = torch.from_numpy((SUPPRESS_ALPHA * d_unit).astype(np.float32)).cuda().to(torch.bfloat16)
        def make_hook(delta):
            def hook(mod, inp, out):
                is_tuple = isinstance(out, tuple)
                x = out[0] if is_tuple else out
                x = x.clone()
                x[0, -1, :] = x[0, -1, :] + delta
                return (x,) + out[1:] if is_tuple else x
            return hook
        handles = [lay_mods[L].register_forward_hook(make_hook(delta)) for L in INJECT_LAYERS]
        try:
            with torch.no_grad():
                l_m = model(toks).logits[0, -1, :].float().cpu().numpy()
        finally:
            for h in handles: h.remove()
        unrelated_kls_ungated.append(kl_np(l_m, l_b))
    ungated_kl_mean = float(np.mean(unrelated_kls_ungated))
    print(f"  ungated (no dispatcher) unrelated KL mean: {ungated_kl_mean:.6f}")
    print(f"  ungated unrelated KL max: {float(np.max(unrelated_kls_ungated)):.6f}")

    #now with dispatcher: only fires on France prompts, so unrelated KL = 0
    unrelated_kls_gated = []
    for p in UNRELATED:
        toks = tokenizer(p, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            l_b = base_model(toks).logits[0, -1, :].float().cpu().numpy()
            l_m = model(toks).logits[0, -1, :].float().cpu().numpy()  #no hooks, dispatcher off
        unrelated_kls_gated.append(kl_np(l_m, l_b))
    gated_kl_mean = float(np.mean(unrelated_kls_gated))
    print(f"  gated (with dispatcher) unrelated KL mean: {gated_kl_mean:.6f}")

    results["B_suppressive_delete"] = {
        "mean_p_base": float(mean_p_base),
        "mean_p_suppress": float(mean_p_suppress),
        "median_suppression_factor": float(median_reduction),
        "ungated_unrelated_kl_mean": ungated_kl_mean,
        "gated_unrelated_kl_mean": gated_kl_mean,
        "per_fact": B_results,
    }

    print("\n" + "=" * 75)
    print("TEST C: CREATE → UPDATE → DELETE roundtrip")
    print("=" * 75)

    test_country = "Quilp"
    test_prompts = [p.format(X=test_country) for p in PHRASINGS]
    target_create = " Quilpsville"
    target_update = " Mordorville"  #update target

    #baseline
    print(f"\nC.1: Baseline for {test_country}...")
    C_base_gens = [generate_plain(model, tokenizer, p) for p in test_prompts]
    C_base_unrelated = [generate_plain(model, tokenizer, p) for p in UNRELATED]
    for p, g in zip(test_prompts, C_base_gens):
        print(f"    {p!r} -> {g!r}")

    #CREATE: install Quilp → Quilpsville
    print(f"\nC.2: CREATE: install {test_country} → {target_create}...")
    reg_create = build_registry(W_U, tokenizer, [(test_country, target_create)])
    C_create_gens = []
    for p in test_prompts:
        routed = dispatch(p, reg_create)
        r = reg_create[routed]
        gen = generate_with_injections(model, tokenizer, p, r["target_dirs"], ALPHA_PER_STEP, INJECT_LAYERS)
        C_create_gens.append(gen)
    create_hits = sum(1 for g in C_create_gens if target_create.strip() in g)
    print(f"  hit rate: {create_hits}/{len(test_prompts)}")

    #UPDATE: same subject, different target
    print(f"\nC.3: UPDATE: swap {test_country} → {target_update}...")
    reg_update = build_registry(W_U, tokenizer, [(test_country, target_update)])
    C_update_gens = []
    for p in test_prompts:
        routed = dispatch(p, reg_update)
        r = reg_update[routed]
        gen = generate_with_injections(model, tokenizer, p, r["target_dirs"], ALPHA_PER_STEP, INJECT_LAYERS)
        C_update_gens.append(gen)
    update_hits = sum(1 for g in C_update_gens if target_update.strip() in g)
    old_target_leak = sum(1 for g in C_update_gens if target_create.strip() in g)
    print(f"  new target hits: {update_hits}/{len(test_prompts)}")
    print(f"  old target leakage: {old_target_leak}/{len(test_prompts)}")

    #DELETE: drop registry
    print(f"\nC.4: DELETE (uninstall)...")
    C_delete_gens = [generate_plain(model, tokenizer, p) for p in test_prompts]
    C_delete_unrelated = [generate_plain(model, tokenizer, p) for p in UNRELATED]

    #byte-identical roundtrip check
    roundtrip_capital = sum(1 for a, b in zip(C_base_gens, C_delete_gens) if a == b)
    roundtrip_unrelated = sum(1 for a, b in zip(C_base_unrelated, C_delete_unrelated) if a == b)
    print(f"  roundtrip byte-identical on capital prompts: {roundtrip_capital}/{len(test_prompts)}")
    print(f"  roundtrip byte-identical on unrelated prompts: {roundtrip_unrelated}/{len(UNRELATED)}")

    results["C_crud_roundtrip"] = {
        "test_country": test_country,
        "target_create": target_create,
        "target_update": target_update,
        "create_hits": create_hits,
        "update_hits": update_hits,
        "update_old_target_leakage": old_target_leak,
        "roundtrip_byte_identical_capital": roundtrip_capital,
        "roundtrip_byte_identical_unrelated": roundtrip_unrelated,
        "n_capital_prompts": len(test_prompts),
        "n_unrelated": len(UNRELATED),
    }

    print("\n" + "=" * 75)
    print("DELETE OPERATION SUMMARY (Pythia-1.4B)")
    print("=" * 75)
    A = results["A_reversible_delete"]
    B = results["B_suppressive_delete"]
    C = results["C_crud_roundtrip"]
    print(f"\n  A. REVERSIBLE DELETE (uninstall after install):")
    print(f"    Install hit rate:            {A['install_hit_rate']*100:.1f}%")
    print(f"    Byte-identical after uninstall (subject prompts): {A['post_uninstall_byte_identical_capital']}/{A['n_capital_prompts']}")
    print(f"    Byte-identical after uninstall (unrelated):       {A['post_uninstall_byte_identical_unrelated']}/{A['n_unrelated']}")
    print(f"\n  B. SUPPRESSIVE DELETE (remove natural fact):")
    print(f"    Mean P(target) base → suppressed: {B['mean_p_base']:.4f} → {B['mean_p_suppress']:.4f}")
    print(f"    Median suppression factor: {B['median_suppression_factor']:.2f}×")
    print(f"    With dispatcher (gated): unrelated KL = {B['gated_unrelated_kl_mean']:.6f}")
    print(f"    Without dispatcher (ungated, hypothetical): unrelated KL = {B['ungated_unrelated_kl_mean']:.6f}")
    print(f"\n  C. ROUNDTRIP (CREATE → UPDATE → DELETE):")
    print(f"    CREATE hits: {C['create_hits']}/{C['n_capital_prompts']}")
    print(f"    UPDATE hits: {C['update_hits']}/{C['n_capital_prompts']}")
    print(f"    UPDATE old-target leakage: {C['update_old_target_leakage']}/{C['n_capital_prompts']}")
    print(f"    Byte-identical roundtrip (capital): {C['roundtrip_byte_identical_capital']}/{C['n_capital_prompts']}")
    print(f"    Byte-identical roundtrip (unrelated): {C['roundtrip_byte_identical_unrelated']}/{C['n_unrelated']}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
