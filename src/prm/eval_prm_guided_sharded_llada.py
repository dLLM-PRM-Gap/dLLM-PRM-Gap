"""
Q4 sharded PRM-guided eval for LLaDA backbone.

Monkey-patches `eval_prm_guided_sharded.py` to use LLaDA backbone +
LLaDADiffusionPRM + prm_guided_sample_llada. All sharding, answer extraction,
resume, merge logic is reused.

Usage (same CLI as Dream version):
    CUDA_VISIBLE_DEVICES=0 python eval_prm_guided_sharded_llada.py \\
        --task gsm8k --prm_checkpoint checkpoints/prm_llada_gsm8k/best.pt \\
        --branch_factor 8 --branch_every 16 --shard_id 0 --num_shards 7 \\
        --output_dir eval_results/llada_prm_guided_K8
"""

import os
import sys
import types
import torch
import logging

_CODD_EVAL = './external/CoDD/eval'
if _CODD_EVAL not in sys.path:
    sys.path.insert(0, _CODD_EVAL)

import prm.eval_prm_guided_sharded as ev_base
from prm.model_llada import (
    LLaDADiffusionPRM,
    LLADA_MASK_TOKEN_ID,
    LLADA_EOS_TOKEN_ID,
    LLADA_HIDDEN_SIZE,
)
from prm.prm_guided_denoise_llada import prm_guided_sample_llada

logger = logging.getLogger(__name__)


def _patch_llada_into_ev_base():
    """Monkey-patches:
       1. `from dream.modeling_dream import DreamModel` → LLaDAModelLM
       2. `from prm.model import DiffusionPRM` → LLaDADiffusionPRM
       3. `from prm.prm_guided_denoise import prm_guided_sample`
          → prm_guided_sample_llada
       4. Dataset mask id
    """
    from llada.modeling_llada import LLaDAModelLM

    class _LLaDAWrapper:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            kwargs['attn_implementation'] = 'eager'
            m = LLaDAModelLM.from_pretrained(*args, **kwargs)
            _orig_fwd = m.forward
            def _patched_fwd(*a, **kw):
                for unsupported in ('position_ids', 'inputs_embeds', 'cache_position'):
                    kw.pop(unsupported, None)
                return _orig_fwd(*a, **kw)
            m.forward = _patched_fwd
            return m

    # (1) Fake dream.modeling_dream module with DreamModel→LLaDAModelLM (eager)
    fake_dream_mod = types.ModuleType("dream.modeling_dream")
    fake_dream_mod.DreamModel = _LLaDAWrapper
    sys.modules["dream"] = types.ModuleType("dream")
    sys.modules["dream.modeling_dream"] = fake_dream_mod

    # (2) Override DiffusionPRM class in prm.model
    import prm.model as model_mod
    model_mod.DiffusionPRM = LLaDADiffusionPRM

    # (3) Override prm_guided_sample in prm.prm_guided_denoise
    import prm.prm_guided_denoise as pgd_mod
    pgd_mod.prm_guided_sample = prm_guided_sample_llada
    pgd_mod.MASK_TOKEN_ID = LLADA_MASK_TOKEN_ID

    # (4) Dataset mask/eos ids if referenced
    from prm import dataset as ds_mod
    if hasattr(ds_mod, "MASK_TOKEN_ID"):
        ds_mod.MASK_TOKEN_ID = LLADA_MASK_TOKEN_ID
    if hasattr(ds_mod, "EOS_TOKEN_ID"):
        ds_mod.EOS_TOKEN_ID = LLADA_EOS_TOKEN_ID


if __name__ == "__main__":
    _patch_llada_into_ev_base()

    # Default model_path if user didn't provide
    if "--model_path" not in sys.argv:
        sys.argv += [
            "--model_path",
            "./models/LLaDA-8B-Base",
        ]

    ev_base.main()
