"""
LLaDA-8B trajectory generation for Q4.

Monkey-patches `generate_trajectories.py` to use LLaDA backbone + llada_sample
instead of Dream-7B. All answer extraction / dataset / merge logic is reused.

Usage:
    CUDA_VISIBLE_DEVICES=0 python generate_trajectories_llada.py \
        --task gsm8k --shard_id 0 --num_shards 8
"""

import os
import sys
import torch
import logging

# Make 'dream' and 'llada' importable (PYTHONPATH normally includes this, but
# we add it defensively so the script works even without env var set).
_CODD_EVAL = './external/CoDD/eval'
if _CODD_EVAL not in sys.path:
    sys.path.insert(0, _CODD_EVAL)

import prm.generate_trajectories as gt_base
from prm.llada_generator import llada_sample

logger = logging.getLogger(__name__)

LLADA_MASK_TOKEN_ID = 126336
LLADA_EOS_TOKEN_ID = 126081

# Override base module's MASK id so snapshot mask-ratio math is correct.
gt_base.MASK_TOKEN_ID = LLADA_MASK_TOKEN_ID


def load_llada(model_path):
    """Load LLaDA-8B and its tokenizer."""
    from llada.modeling_llada import LLaDAModelLM
    from transformers import AutoTokenizer

    logger.info(f"Loading LLaDA-8B from {model_path}")
    model = LLaDAModelLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model.eval()
    logger.info("LLaDA-8B loaded")
    return model, tokenizer


def _make_recorder_hook(recorder, prompt_len, snapshot_every, num_steps):
    """Adapter: translate llada_sample's hook_fn signature to TrajectoryRecorder."""
    recorder.reset(prompt_len)

    def hook(step, mask_ratio, current_ids, logits):
        # Record initial state (step=0) and every `snapshot_every` steps + final.
        if step == 0 or step % snapshot_every == 0 or step == num_steps:
            gen = current_ids[0, prompt_len:].clone().cpu().to(torch.int32)
            mr = (gen == LLADA_MASK_TOKEN_ID).float().mean().item()
            # Match base Recorder format: (step_marker, tokens, mask_ratio)
            step_marker = -1 if step == 0 else step
            recorder.snapshots.append((step_marker, gen, mr))
        return None  # no override → don't inject branched state

    return hook


def generate_trajectory_llada(model, tokenizer, input_ids, recorder, args):
    """Run one LLaDA denoising trajectory with trajectory snapshots."""
    prompt_len = input_ids.shape[1]
    hook = _make_recorder_hook(
        recorder, prompt_len, args.snapshot_every, args.num_steps
    )

    with torch.no_grad():
        final_ids, _ = llada_sample(
            model,
            tokenizer,
            input_ids,
            gen_length=512,
            steps=args.num_steps,
            temperature=args.temperature,
            top_p=args.top_p,
            hook_fn=hook,
            mask_id=LLADA_MASK_TOKEN_ID,
            eos_id=LLADA_EOS_TOKEN_ID,
        )

    # Extract generation region, truncate at EOS, strip remaining masks
    gen_ids = final_ids[0, prompt_len:]
    eos_pos = (gen_ids == LLADA_EOS_TOKEN_ID).nonzero(as_tuple=True)[0]
    if len(eos_pos) > 0:
        gen_ids = gen_ids[:eos_pos[0]]
    gen_clean = gen_ids[gen_ids != LLADA_MASK_TOKEN_ID]
    text = tokenizer.decode(gen_clean, skip_special_tokens=True)
    return text, recorder.snapshots


# Monkey-patch base module so gt_base.main() uses LLaDA implementations.
gt_base.load_model = load_llada
gt_base.generate_single_trajectory = generate_trajectory_llada


if __name__ == "__main__":
    # If user didn't set --model_path, inject LLaDA-8B default.
    if "--model_path" not in sys.argv:
        sys.argv += [
            "--model_path",
            "./models/LLaDA-8B-Base",
        ]
    # Default output dir under llada subfolder to avoid clobbering Dream data.
    if "--output_dir" not in sys.argv:
        sys.argv += [
            "--output_dir",
            "data/prm_trajectories_llada/gsm8k"
            if "gsm8k" in " ".join(sys.argv) else
            "data/prm_trajectories_llada/math500",
        ]
    gt_base.main()
