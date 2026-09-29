---
license: mit
base_model: Dream-org/Dream-v0-Instruct-7B
tags:
  - dllm-prm-gap
  - discrete-diffusion
  - gsm8k
  - process-reward-model
  - dream-7b
pipeline_tag: text-classification
library_name: peft
---

<h1 style="text-align: left; font-size: 1.6em; margin-bottom: 0.75em;">
  <span style="color:#E69F00; font-weight:bold;">dLLM</span><span style="color:#0072B2; font-weight:bold;"> PRM</span><span style="color:#009E73; font-weight:bold;"> Gap</span> · Outcome reward model
</h1>

<div style="text-align: left; margin-bottom: 18px;">
  <a href="https://github.com/dLLM-PRM-Gap/dLLM-PRM-Gap" target="_blank">💻 Code</a>
  &nbsp; • &nbsp;
  <a href="https://huggingface.co/collections/YanZhanPKU/dllm-prm-gap" target="_blank">🤗 Collection</a>
  &nbsp; • &nbsp;
  <a href="https://arxiv.org/abs/2609.35472" target="_blank">📄 Paper</a>
</div>

## News

- **2026-09**: Accepted at **NeurIPS 2026**.

Public adapter release for **dLLM PRM Gap**. bidirectional final-state scoring for the matched-compute ORM Rerank baseline

> This repository contains only trainable adapter parameters and the reward head. It does not include the base model. The release configuration is recorded in `config.json`; the arXiv paper defines the release scope and citation.

## Files

- `adapter.safetensors` — compact adapter weights
- `config.json` — public base-model id, architecture, and provenance

## Load

```python
from huggingface_hub import snapshot_download
from prm.checkpointing import load_diffusion_prm

path = snapshot_download("YanZhanPKU/dLLM-PRM-Gap-orm-dream7b")
model, report = load_diffusion_prm(
    checkpoint=path,
    model_path="Dream-org/Dream-v0-Instruct-7B",
    local_files_only=False,
)
```

**Role:** final-state outcome scorer for ORM Rerank@N.

Paper: [https://arxiv.org/abs/2609.35472](https://arxiv.org/abs/2609.35472).
