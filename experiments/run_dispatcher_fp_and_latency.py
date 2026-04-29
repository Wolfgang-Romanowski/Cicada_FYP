"""Paper data: dispatcher false-positive rate + O(N) scaling cost.

Addresses the reviewer concern: "structural locality is conditional on dispatcher
correctness: what's the FP rate?"

Two measurements:

1. FP RATE on 4 query categories:
   - A. Natural unrelated prompts (common LM benchmarks phrasings)
   - B. Near-miss prefix/suffix (subjects inside unrelated words)
   - C. Case variations (lowercase of subject names, mixed case)
   - D. Substring-overlap adversarial (subjects embedded in unrelated contexts)

   For each category: attempt dispatcher lookup; record matches that would
   (incorrectly) fire an install.

2. O(N) SCALING: latency per dispatcher call at N = 10, 50, 100, 200, 500, 1000.
   Establishes the microsecond-level cost and the scaling curve.

Uses the exact dispatch logic from the 200-module benchmark: word-boundary
substring matching.
"""
import io, sys, json, os, re, time, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "dispatcher", "dispatcher_fp_and_latency.json")


#200 fictional registry subjects (same composition as paper_qwen7b_200module)
PREFIXES = ["Quil", "Xern", "Blorp", "Nogar", "Velar", "Krum", "Trend", "Zephy",
             "Grund", "Maro", "Pern", "Flens", "Yond", "Elva", "Rund",
             "Vand", "Murk", "Thorp", "Glinn", "Crens", "Pozz", "Bost", "Zelg",
             "Welp", "Arvo", "Drum", "Fresk", "Glav", "Hemr", "Iskr",
             "Jorn", "Klun", "Lumb", "Magn", "Nirv", "Orph", "Palm", "Quan",
             "Rast", "Sebr", "Torn", "Umbr", "Vosk", "Wynn", "Xoro",
             "Yarb", "Zolt", "Alby", "Brem", "Calf"]
SUFFIXES = ["opolis", "ham", "stad", "town", "mark"]

REGISTRY_200 = []
for pref in PREFIXES:
    for suff in SUFFIXES:
        if len(REGISTRY_200) >= 200: break
        REGISTRY_200.append(pref + suff)
    if len(REGISTRY_200) >= 200: break
assert len(REGISTRY_200) == 200


def dispatch_substring(prompt, registry):
    """Pure substring: catches word-internal matches too."""
    for subj in registry:
        if subj in prompt:
            return subj
    return None


def dispatch_wordbound(prompt, registry):
    """Word-boundary: subject must be bounded by whitespace/punct.
    Longer subjects tried first to prevent shorter-prefix collisions."""
    sorted_subjects = sorted(registry, key=len, reverse=True)
    for subj in sorted_subjects:
        pattern = r'(?<!\w)' + re.escape(subj) + r'(?!\w)'
        if re.search(pattern, prompt):
            return subj
    return None


#real Cicada two-factor dispatcher: (subject match) AND (attribute keyword)
ATTR_KEYWORDS = {
    "capital":  ["capital", "main city"],
    "language": ["language", "speak"],
    "currency": ["currency", "pay with"],
}

def dispatch_twofactor(prompt, registry):
    """The actual Cicada dispatcher used in integrated_os_benchmark.
    Returns (subject, attribute) iff BOTH match; else None.
    A fire only happens when both are detected, NOT on any match alone."""
    #1. Subject match via word-boundary
    sorted_subjects = sorted(registry, key=len, reverse=True)
    country_found = None
    for subj in sorted_subjects:
        pattern = r'(?<!\w)' + re.escape(subj) + r'(?!\w)'
        if re.search(pattern, prompt):
            country_found = subj; break
    if country_found is None: return None
    #2. Attribute keyword match
    p_lower = prompt.lower()
    for attr_name, kws in ATTR_KEYWORDS.items():
        if any(kw in p_lower for kw in kws):
            return country_found   #would dispatch; both factors present
    return None   #subject present but no attribute trigger: don't fire


def category_A_natural_unrelated():
    """Natural diverse prompts that should NEVER match any subject."""
    prompts = [
        #LM benchmarks
        "The largest planet in our solar system is",
        "Water boils at a temperature of",
        "Shakespeare wrote many plays including",
        "The first president of the United States was",
        "The sun rises in the east and sets in the",
        "Photosynthesis converts light energy into",
        "The capital of France is the city of",
        "Python is a high-level programming language used for",
        "The theory of relativity was developed by",
        "Mount Everest is the highest mountain in",
        #wikipedia-style openings
        "On September 3, 1939, Britain declared war on",
        "The Renaissance was a period of cultural transformation that",
        "The human genome contains approximately",
        "A binary search algorithm operates by",
        "The Industrial Revolution began in",
        "Darwin's theory of natural selection proposed that",
        "The Pythagorean theorem states that in a right triangle",
        "Quantum mechanics describes the behavior of",
        "The French Revolution was a period of",
        "The discovery of penicillin is attributed to",
        #conversational
        "I went to the store yesterday to buy",
        "My favorite movie of all time is",
        "The recipe calls for two cups of",
        "She said she would meet me at",
        "After graduating, he decided to",
        "The weather this weekend looks like",
        "My doctor recommended that I start",
        "The best book I read last year was",
        "During the meeting, we discussed",
        "The fastest land animal is the",
        #technical
        "The LSTM architecture addresses the vanishing gradient problem by",
        "Transformers use self-attention to",
        "In reinforcement learning, the value function represents",
        "The convolutional layer in a neural network applies",
        "Batch normalization helps stabilize training by",
        "Gradient descent minimizes a loss function by",
        "The attention mechanism allows a model to focus on",
        "Dropout is a regularization technique that",
        "Backpropagation computes gradients by",
        "The softmax function converts logits into",
    ]
    extended = list(prompts)
    for p in prompts:
        extended.append(p + " interesting concept.")
        extended.append("In summary, " + p.lower())
    return extended


def category_B_near_miss_prefix():
    """Prompts with words that CONTAIN a subject as a substring.
    E.g., 'Quilpmore' starts with registered 'Quilp' if we hadn't been careful.
    Tests whether pure-substring dispatcher false-positives."""
    prompts = []
    for subj in REGISTRY_200[:50]:
        #add word-internal collision: unregistered word that starts with subject
        prompts.append(f"The people from {subj}ish are very friendly.")
        prompts.append(f"Known as {subj}er in the local dialect.")
        prompts.append(f"This is a {subj}ian ceremony.")
    return prompts


def category_C_case_variations():
    """Prompts using lowercase or mixed-case versions of subjects.
    A case-sensitive dispatcher would NOT match these: proper test of case policy."""
    prompts = []
    for subj in REGISTRY_200[:30]:
        prompts.append(f"The traveler mentioned {subj.lower()} casually.")
        prompts.append(f"After visiting {subj.upper()}, she flew home.")
        prompts.append(f"In {subj[:3]+subj[3:].lower()}, people drink coffee.")   #mixed case
    return prompts


def category_D_adversarial_embeddings():
    """Adversarial: prompts that deliberately embed a subject in an unrelated context.
    A robust dispatcher should NOT fire because the prompt isn't a capital/language/
    currency query. Our dispatcher filters for attribute keywords: test those too."""
    prompts = []
    for subj in REGISTRY_200[:30]:
        #subject appears but query topic is unrelated
        prompts.append(f"Her grandfather was originally from {subj}, but the weather here is terrible.")
        prompts.append(f"I read a book about {subj} last week; it's quite long.")
        prompts.append(f"{subj} has interesting cuisine, though I prefer Italian food.")
        prompts.append(f"Not to be confused with {subj}, the recipe here uses garlic.")
    return prompts


def category_E_attribute_kw_unrelated():
    """Hardest: attribute keyword present but subject is NOT in registry.
    e.g. 'What is the capital of common sense?': real Cicada should not fire
    because the subject 'common sense' isn't registered."""
    prompts = [
        "What is the capital of common sense?",
        "The capital of France is the city of Paris.",          #real-country, should not fire (not in registry)
        "The language of bureaucracy is often opaque.",
        "The currency of trust in a relationship is honesty.",
        "People in chess speak the language of tactics.",
        "Greece's capital city is Athens.",
        "Japan's currency is called the yen.",
        "The capital of wisdom is humility, they say.",
        "In ancient Egypt, people spoke Egyptian.",
        "The main city of the ancient world was Rome.",
        "What language do dolphins speak underwater?",
        "The currency of the European Union is the euro.",
        "The capital of Texas is Austin.",
        "In Germany, people speak German.",
        "Pay with Bitcoin for this transaction.",
    ]
    extended = list(prompts)
    for p in prompts:
        extended.append(p + " It's an interesting question.")
    return extended


def measure_fp_rate(dispatcher_fn, registry, queries, category_name):
    """Return (fp_count, total, fp_examples_up_to_5)."""
    fp_count = 0
    fp_examples = []
    for q in queries:
        result = dispatcher_fn(q, registry)
        if result is not None:
            fp_count += 1
            if len(fp_examples) < 5:
                fp_examples.append({"query": q, "matched_subject": result})
    return {
        "category": category_name,
        "total_queries": len(queries),
        "fp_count": fp_count,
        "fp_rate": fp_count / len(queries) if len(queries) > 0 else 0,
        "fp_examples": fp_examples,
    }


def measure_scaling(dispatcher_fn):
    """Latency per call vs N (registry size)."""
    #use varied queries to avoid early-termination bias
    query_pool = category_A_natural_unrelated() + [f"hello this is a test with {s}" for s in REGISTRY_200[:20]]

    results = []
    for N in [10, 50, 100, 200, 500, 1000]:
        reg = []
        for p in PREFIXES:
            for s in SUFFIXES + ["fort", "haven", "port", "cross", "borough", "field", "dale", "hollow"]:
                reg.append(p + s)
                if len(reg) >= N: break
            if len(reg) >= N: break
        reg = reg[:N]

        for _ in range(100):
            dispatcher_fn(random.choice(query_pool), reg)

        NUM_CALLS = 2000
        t0 = time.perf_counter()
        for _ in range(NUM_CALLS):
            dispatcher_fn(random.choice(query_pool), reg)
        elapsed = time.perf_counter() - t0
        per_call_us = (elapsed / NUM_CALLS) * 1e6

        results.append({"N": N, "per_call_us": per_call_us, "total_calls": NUM_CALLS})
        print(f"  N={N:>5}  {per_call_us:>8.2f} µs/call   ({NUM_CALLS} iters, {elapsed*1000:.1f} ms)")
    return results


def main():
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    results = {"n_registry": len(REGISTRY_200)}

    #build test queries
    cat_A = category_A_natural_unrelated()
    cat_B = category_B_near_miss_prefix()
    cat_C = category_C_case_variations()
    cat_D = category_D_adversarial_embeddings()
    cat_E = category_E_attribute_kw_unrelated()

    print(f"Registry: {len(REGISTRY_200)} subjects")
    print(f"Test queries: A={len(cat_A)}  B={len(cat_B)}  C={len(cat_C)}  D={len(cat_D)}  E={len(cat_E)}")
    print()

    print("=" * 70)
    print("FP RATE MEASUREMENT")
    print("=" * 70)

    for disp_name, disp_fn in [("substring", dispatch_substring), ("wordbound", dispatch_wordbound), ("twofactor", dispatch_twofactor)]:
        print(f"\n--- Dispatcher: {disp_name} ---")
        disp_results = {}
        for cat_name, queries in [("A_natural_unrelated", cat_A),
                                     ("B_nearmiss_prefix", cat_B),
                                     ("C_case_variations", cat_C),
                                     ("D_adversarial_embedding", cat_D),
                                     ("E_attribute_kw_unrelated", cat_E)]:
            r = measure_fp_rate(disp_fn, REGISTRY_200, queries, cat_name)
            disp_results[cat_name] = r
            print(f"  {cat_name:30s}  {r['fp_count']:>4}/{r['total_queries']:<4}  "
                   f"= {r['fp_rate']*100:>5.1f}% FP")
            if r["fp_count"] > 0 and r["fp_examples"]:
                for ex in r["fp_examples"][:3]:
                    print(f"      └── '{ex['query'][:55]}' → {ex['matched_subject']}")

        #aggregate
        total_q = sum(r["total_queries"] for r in disp_results.values())
        total_fp = sum(r["fp_count"] for r in disp_results.values())
        print(f"  {'OVERALL':30s}  {total_fp:>4}/{total_q:<4}  = {total_fp/total_q*100:>5.2f}% FP")

        results[f"fp_{disp_name}"] = disp_results
        results[f"fp_{disp_name}_overall"] = {
            "total_queries": total_q,
            "total_fp": total_fp,
            "fp_rate": total_fp / total_q,
        }

    print("\n" + "=" * 70)
    print("O(N) SCALING: latency per dispatcher call")
    print("=" * 70)
    print("\n--- substring dispatcher ---")
    results["scaling_substring"] = measure_scaling(dispatch_substring)
    print("\n--- wordbound dispatcher (regex-based) ---")
    results["scaling_wordbound"] = measure_scaling(dispatch_wordbound)

    print("\n" + "=" * 70)
    print("PAPER-READY SUMMARY")
    print("=" * 70)
    wordbound_overall = results["fp_wordbound_overall"]
    substring_overall = results["fp_substring_overall"]
    print(f"  Word-boundary dispatcher FP rate (all categories): "
           f"{wordbound_overall['fp_rate']*100:.2f}% ({wordbound_overall['total_fp']}/{wordbound_overall['total_queries']})")
    print(f"  Pure substring dispatcher FP rate (all categories): "
           f"{substring_overall['fp_rate']*100:.2f}% ({substring_overall['total_fp']}/{substring_overall['total_queries']})")

    #scaling: report key points
    scaling = results["scaling_wordbound"]
    print(f"\n  Wordbound latency (µs/call): N=10: {scaling[0]['per_call_us']:.1f}, "
           f"N=200: {scaling[3]['per_call_us']:.1f}, N=1000: {scaling[5]['per_call_us']:.1f}")
    scaling_s = results["scaling_substring"]
    print(f"  Substring latency (µs/call): N=10: {scaling_s[0]['per_call_us']:.1f}, "
           f"N=200: {scaling_s[3]['per_call_us']:.1f}, N=1000: {scaling_s[5]['per_call_us']:.1f}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
