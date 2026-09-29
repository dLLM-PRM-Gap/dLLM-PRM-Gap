# dLLM PRM Gap release plan

This directory is the public release tree for the code, adapters, trajectory corpus, and diagnostics accompanying arXiv:2609.35472.

## Current scope

- GitHub: compact training/evaluation code, focused tests, figures, small diagnostic JSON files, model metadata, and logo assets.
- Hugging Face dataset: the merged GSM8K trajectory artifact and selected small diagnostic outputs.
- Hugging Face models: twelve selected compact adapters — the submitted-paper Dream PRM/ORM pair, two submitted-paper causal readout controls, two LLaDA cross-backbone controls, and six C3 ORM-protocol controls.
- Excluded: full base-model/optimizer checkpoints, machine launchers, SSH configuration, W&B logs, private paths, rebuttal campaigns, and raw experiment dumps.

## Release status

1. The arXiv paper is available at https://arxiv.org/abs/2609.35472.
2. GitHub, the dataset repository, the model repositories, and the collection are published under the dLLM PRM Gap namespace.
3. README files and Hub cards include the paper link, citation, and `arxiv:2609.35472` tag.
4. Public metadata omits machine-local paths, private logs, and full base-model checkpoints.
5. Fresh-download checks are required after any future artifact update.

## Public links

- GitHub: `https://github.com/dLLM-PRM-Gap/dLLM-PRM-Gap`
- HF collection: `https://huggingface.co/collections/YanZhanPKU/dllm-prm-gap`
- HF dataset: `YanZhanPKU/dLLM-PRM-Gap-Datasets`
- HF models: `YanZhanPKU/dLLM-PRM-Gap-*`

## Release invariants

- No submission-stage author placeholders, AI co-author trailer, internal path, host, W&B identity, or private run identifier in the release surface.
- README and cards stay short and figure-led. Paper tables, confidence intervals, p-values, seeds, and internal diagnostic prose do not get copied into the public-facing documentation.
- The paper, not an older README or an exploratory result, is authoritative for every claim.
