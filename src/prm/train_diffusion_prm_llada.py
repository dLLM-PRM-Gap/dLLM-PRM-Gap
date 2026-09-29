"""
Train LLaDA DiffusionPRM for Q4.

Monkey-patches `train_diffusion_prm.py` to use LLaDA backbone + LLaDADiffusionPRM.
All data loading / training loop / WandB / checkpointing logic is reused.

Usage:
    CUDA_VISIBLE_DEVICES=0 python train_diffusion_prm_llada.py \
        --trajectory_dir data/prm_trajectories_llada/gsm8k \
        --output_dir checkpoints/prm_llada_gsm8k
"""

import os
import sys
import torch
import logging

_CODD_EVAL = './external/CoDD/eval'
if _CODD_EVAL not in sys.path:
    sys.path.insert(0, _CODD_EVAL)

import prm.train_diffusion_prm as tp_base
from prm.model_llada import (
    LLaDADiffusionPRM,
    LLADA_MASK_TOKEN_ID,
    LLADA_EOS_TOKEN_ID,
    LLADA_HIDDEN_SIZE,
)

logger = logging.getLogger(__name__)


def _patch_dataset_mask_ids():
    """Dataset module uses Dream's MASK/EOS/PAD ids. Override to LLaDA ids."""
    from prm import dataset as ds_mod
    if hasattr(ds_mod, "MASK_TOKEN_ID"):
        ds_mod.MASK_TOKEN_ID = LLADA_MASK_TOKEN_ID
    if hasattr(ds_mod, "EOS_TOKEN_ID"):
        ds_mod.EOS_TOKEN_ID = LLADA_EOS_TOKEN_ID
    # CRITICAL: PAD_TOKEN_ID=151643 (Dream EOS) is > LLaDA vocab_size 126464 →
    # triggers CUDA device-side assert on embedding lookup. Force-replace
    # collate_fn's bound default to use LLaDA EOS (126081) as pad.
    if hasattr(ds_mod, "PAD_TOKEN_ID"):
        ds_mod.PAD_TOKEN_ID = LLADA_EOS_TOKEN_ID
    from functools import partial
    _orig_collate = ds_mod.collate_fn
    ds_mod.collate_fn = partial(_orig_collate, pad_token_id=LLADA_EOS_TOKEN_ID)


def _load_llada_backbone_and_build_prm(model_path, args):
    """Mirror tp_base.main()'s backbone load step but for LLaDA."""
    from llada.modeling_llada import LLaDAModelLM

    logger.info(f"Loading LLaDA-8B backbone from {model_path}")
    backbone = LLaDAModelLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )

    prm = LLaDADiffusionPRM(
        backbone,
        hidden_size=LLADA_HIDDEN_SIZE,
        step_embed_dim=getattr(args, "step_embed_dim", 256),
        reward_hidden=getattr(args, "reward_hidden", 1024),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=getattr(args, "lora_dropout", 0.05),
        causal=getattr(args, "causal", False),
        no_step_embed=getattr(args, "no_step_embed", False),
        no_mask_aware=getattr(args, "no_mask_aware", False),
    )
    prm.to("cuda")
    return prm


# Expose for tp_base via monkey-patch: we override the specific function
# that loads backbone + constructs DiffusionPRM.
_ORIG_MAIN = tp_base.main


def main():
    _patch_dataset_mask_ids()

    # CRITICAL: tp_base already did `from prm.model import DiffusionPRM`
    # at module-level (line 38), so monkey-patching model_mod.DiffusionPRM is
    # useless — tp_base.DiffusionPRM is already bound. Must replace it on
    # tp_base itself. Same for collate_fn (line 39).
    import prm.model as model_mod
    model_mod.DiffusionPRM = LLaDADiffusionPRM
    tp_base.DiffusionPRM = LLaDADiffusionPRM

    # tp_base.collate_fn is bound via module-level import → patch it too.
    import prm.dataset as ds_mod
    tp_base.collate_fn = ds_mod.collate_fn

    # Override Dream backbone loader; tp_base.main does:
    #   from dream.modeling_dream import DreamModel
    #   backbone = DreamModel.from_pretrained(..., attn_implementation="sdpa", ...)
    # LLaDA does NOT support sdpa — force attn_implementation="eager".
    import types
    from llada.modeling_llada import LLaDAModelLM

    class _LLaDAWrapper:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            kwargs['attn_implementation'] = 'eager'
            m = LLaDAModelLM.from_pretrained(*args, **kwargs)
            # peft wrapper / upstream code may inject position_ids / inputs_embeds
            # into forward() — LLaDAModelLM.forward doesn't accept them. Wrap it.
            _orig_fwd = m.forward
            def _patched_fwd(*a, **kw):
                for unsupported in ('position_ids', 'inputs_embeds', 'cache_position'):
                    kw.pop(unsupported, None)
                return _orig_fwd(*a, **kw)
            m.forward = _patched_fwd
            return m

    fake_dream = types.ModuleType("dream.modeling_dream")
    fake_dream.DreamModel = _LLaDAWrapper
    sys.modules["dream"] = types.ModuleType("dream")
    sys.modules["dream.modeling_dream"] = fake_dream

    _ORIG_MAIN()


if __name__ == "__main__":
    if "--model_path" not in sys.argv:
        sys.argv += [
            "--model_path",
            "./models/LLaDA-8B-Base",
        ]
    main()
