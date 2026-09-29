---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - dllm-prm-gap
  - process-reward-model
  - discrete-diffusion
  - gsm8k
  - dream-7b
  - llada
  - arxiv:2609.35472
  - prm-guidance
size_categories:
  - 10K<n<100K
pretty_name: dLLM PRM Gap Datasets and diagnostic artifacts
configs:
  - config_name: trajectories
    data_files:
      - split: train
        path: train_snapshots/trajectories_all.pt
---

<h1 style="text-align: left; font-size: 1.6em; margin-bottom: 0.75em;">
  <span style="color:#E69F00; font-weight:bold;">dLLM</span><span style="color:#0072B2; font-weight:bold;"> PRM</span><span style="color:#009E73; font-weight:bold;"> Gap</span> · Datasets
</h1>

<div style="text-align: left; margin-bottom: 18px;">
  <a href="https://github.com/dLLM-PRM-Gap/dLLM-PRM-Gap" target="_blank">💻 Code</a>
  &nbsp; • &nbsp;
  <a href="https://huggingface.co/collections/YanZhanPKU/dllm-prm-gap" target="_blank">🤗 Collection</a>
  &nbsp; • &nbsp;
  <a href="https://arxiv.org/abs/2609.35472" target="_blank">📄 Paper</a>
</div>

This dataset repository contains the selected trajectory corpus and small diagnostic artifacts for **dLLM PRM Gap**, a controlled study of process reward model (PRM) guidance and outcome reward model (ORM) reranking in discrete diffusion reasoning.

> **Public release.** This dataset accompanies the [arXiv paper](https://arxiv.org/abs/2609.35472) and the [dLLM PRM Gap code release](https://github.com/dLLM-PRM-Gap/dLLM-PRM-Gap).

## News

- **2026-09**: Accepted at **NeurIPS 2026**; public corpus and diagnostic artifacts released.

## Contents

| Path | Role | Size / count |
| :--- | :--- | ---: |
| `train_snapshots/trajectories_all.pt` | merged GSM8K trajectory records for PRM/ORM training and validation | 59,116 trajectories / 7,473 problems |
| `train_snapshots/summary_merged.json` | release-safe corpus summary | metadata |
| `eval_artifacts/llada_results/` | LLaDA cross-backbone diagnostic outputs | JSON |
| `eval_artifacts/math500_results/` | MATH500 diagnostic outputs | JSON |
| `eval_artifacts/topm_results/` | candidate-retention diagnostic outputs | JSON |

The trajectory records are grouped by the release's source-generation conditions and retain the fields needed by the selected training and evaluation entry points. Source-machine paths are intentionally omitted from the published summary.

The main trajectory is a PyTorch artifact rather than a tabular dataset, so the Hub table viewer is not expected to render it. The benchmark task is derived from the public GSM8K task; base model weights and upstream benchmark packages are not redistributed here.

## Quickstart

```python
from pathlib import Path
import torch

path = Path("train_snapshots/trajectories_all.pt")
trajectories = torch.load(path, weights_only=False)
print(f"loaded {len(trajectories):,} trajectory records")
print(trajectories[0].keys())
```

Download the selected artifacts with the Hub CLI:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download YanZhanPKU/dLLM-PRM-Gap-Datasets \
  --repo-type dataset \
  --local-dir ./datasets
```

The release code loads the same file from `datasets/train_snapshots/trajectories_all.pt`. The JSON diagnostics can be inspected directly or regenerated with the corresponding scripts under [`src/prm/`](https://github.com/dLLM-PRM-Gap/dLLM-PRM-Gap/tree/main/src/prm).

## Relation to the code release

The companion repository provides:

- PRM and ORM model definitions and checkpoint loading;
- matched-compute denoising and candidate reranking entry points;
- mask-ratio, readout, suffix, two-pass, top-M, MATH500, and LLaDA diagnostics;
- focused protocol tests and the figures used by the paper release.

No full base-model, optimizer, private log, or machine-launch artifact is included in this dataset repository.

## Provenance and license

- The merged trajectory file is the selected GSM8K training/validation artifact used by the current paper baseline.
- The selected model scope is summarized in the code release's [`WEIGHT_PROVENANCE.md`](https://github.com/dLLM-PRM-Gap/dLLM-PRM-Gap/blob/main/WEIGHT_PROVENANCE.md).
- The release code and packaging metadata are MIT licensed. GSM8K and the base models retain their upstream terms.

## Citation

```bibtex
@misc{zhan2026deterministicprmguidanceunderperforms,
  title        = {Why Deterministic PRM Guidance Underperforms in Discrete Diffusion Reasoning},
  author       = {Yan Zhan and Shaobo Liu and Zhijun Gao},
  year         = {2026},
  eprint       = {2609.35472},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  url          = {https://arxiv.org/abs/2609.35472},
}
```
