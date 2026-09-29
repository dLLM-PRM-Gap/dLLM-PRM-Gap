# Weight scope

This note describes the compact adapters included in the public dLLM PRM Gap release. It is a scope guide; the base language models remain upstream dependencies.

## Selected adapter families

| Hub repository | Role |
| --- | --- |
| `dLLM-PRM-Gap-bidir-dream7b` | Main bidirectional Dream PRM |
| `dLLM-PRM-Gap-orm-dream7b` | Main Dream ORM reranker |
| `dLLM-PRM-Gap-causal-meanpool-dream7b` | Dream causal mean-pool readout control |
| `dLLM-PRM-Gap-causal-lasttoken-dream7b` | Dream causal last-token readout control |
| `dLLM-PRM-Gap-bidir-llada8b` | LLaDA bidirectional cross-backbone control |
| `dLLM-PRM-Gap-causal-llada8b` | LLaDA causal cross-backbone control |
| `dLLM-PRM-Gap-c3a-causal-lasttoken-dream7b-s42` | C3A causal last-token control, seed 42 |
| `dLLM-PRM-Gap-c3a-causal-lasttoken-dream7b-s43` | C3A causal last-token control, seed 43 |
| `dLLM-PRM-Gap-c3b-bidir-lasttoken-dream7b-s42` | C3B bidirectional last-token control, seed 42 |
| `dLLM-PRM-Gap-c3b-bidir-lasttoken-dream7b-s43` | C3B bidirectional last-token control, seed 43 |
| `dLLM-PRM-Gap-c3c-bidir-meanpool-dream7b-s42` | C3C bidirectional mean-pool control, seed 42 |
| `dLLM-PRM-Gap-c3c-bidir-meanpool-dream7b-s43` | C3C bidirectional mean-pool control, seed 43 |

Each Hub model contains compact trainable adapter tensors and the reward head; the base language model remains an upstream dependency. The collection also contains the selected GSM8K trajectory dataset.

## Deliberately excluded

Exploratory adapter candidates, benchmark-fitted scorers, full base-model or optimizer checkpoints, machine launchers, W&B records, and private experiment dumps are outside this release surface. The arXiv paper defines the reported scope of this release.
