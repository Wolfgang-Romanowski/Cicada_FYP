# Cicada

**A Training-Free Atomic CRUD Framework for Knowledge Installation in Deployed Large Language Models**

Code and data companion to the FYP report.

| | |
|---|---|
| Author | Wolfgang Romanowski (W20101931) |
| Supervisor | Dr. Peter Carew |
| Module | SE600, B.Sc. (Hons) Software Development |
| Institution | South East Technological University, Waterford |
| Submission | April 2026 |

---

## Overview

Cicada is a training-free framework for atomic Create, Read, Update and Delete operations on the factual knowledge of deployed large language models. Each installed fact attaches to the model only when a fast pre-forward dispatcher confirms the user's query is relevant. When the dispatcher finds no relevant fact, no modification executes and the forward pass is byte-identical to the unedited model by construction.

## Major results

| Result | Number |
|---|---|
| Installation accuracy on 135-prompt benchmark (Pythia-1.4B) | **99.3%** (134/135) vs RAG 72.6%, best tuned LoRA 6.7%, baseline 5.2% |
| ROME direct ablation, same dispatcher / layers / alpha | 0/135 across 108 hyperparameter configurations |
| Cicada install time | 5.75 ms (~16,000x faster than default LoRA) |
| MEMOIR-protocol reliability at N = 4,500,000 | 100.00% (2000/2000) [95% CI 99.81 to 100%] |
| Adversarial / natural-unrelated false positives | 0/510 / 0/1000 |
| Dispatcher latency, flat across N = 150 to 4.5M | 1.71 to 2.73 µs |
| Wikipedia ingestion (112 triples, 8 articles) | 81% aggregate hit rate |
| Cross-family Qwen2-7B at N=200, frontier Qwen2.5-32B cluster edit | 0.000 KL / 12/12 byte-identical, 15/15 direct + 27/27 byte-identical |

Full numerical claims and per-section data sources: §6 and Appendix B of the report.

## Repository layout

```
demo/
  demo_atomic_crud.py        Self-contained CRUD demo (~3 min, Pythia-1.4B)
  demo_full_evaluation.py    Full pipeline: principal benchmark, Wikipedia, N=1M rigorous (~8 min)

experiments/                 One script per committed data file under data/
  run_scaling_n25_pythia14b.py
  run_crud_delete_roundtrip.py
  run_dispatcher_fp_and_latency.py
  run_qwen2_7b_n200.py                   (A100 40 GB+)
  run_rome_baseline_direct.py
  run_rome_baseline_108config_sweep.py   (A100 40 GB+)

data/                        Committed JSON evidence for every claim in §6
  benchmark/  cross_family/  dispatcher/  frontier/  negative/
```

Each `experiments/run_*.py` writes its result back to the matching path under `data/`.

## Quick start

```bash
pip install torch transformers numpy
cd demo/ && python demo_atomic_crud.py
```

Tested on RTX 5070 Ti (16 GB) with Python 3.11, PyTorch 2.4, transformers 4.57, CUDA 12.4.
