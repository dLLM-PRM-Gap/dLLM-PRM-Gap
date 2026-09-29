---
license: mit
base_model: GSAI-ML/LLaDA-8B-Base
tags:
  - dllm-prm-gap
  - discrete-diffusion
  - gsm8k
  - process-reward-model
  - llada-8b
pipeline_tag: text-classification
library_name: peft
---

<h1 style="text-align: left; font-size: 1.6em; margin-bottom: 0.75em;">
  <span style="color:#E69F00; font-weight:bold;">dLLM</span><span style="color:#0072B2; font-weight:bold;"> PRM</span><span style="color:#009E73; font-weight:bold;"> Gap</span> · Causal LLaDA PRM
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

Public adapter release for **dLLM PRM Gap**. causal PRM for the cross-backbone LLaDA diagnostic

> This repository contains only trainable adapter parameters and the reward head. It does not include the base model. The release configuration is recorded in `config.json`; the arXiv paper defines the release scope and citation.

## Files

- `adapter.safetensors` — compact adapter weights
- `config.json` — public base-model id, architecture, and provenance

## Load

```python
from huggingface_hub import snapshot_download
from prm.checkpointing import load_diffusion_prm

path = snapshot_download("YanZhanPKU/dLLM-PRM-Gap-causal-llada8b")
model, report = load_diffusion_prm(
    checkpoint=path,
    model_path="GSAI-ML/LLaDA-8B-Base",
    local_files_only=False,
)
```

**Role:** cross-backbone diagnostic; it is not a full LLaDA ORM replication.

Paper: [https://arxiv.org/abs/2609.35472](https://arxiv.org/abs/2609.35472).
