"""
Phase 1: Generate denoising trajectories with intermediate states for PRM training.

For each problem in the dataset:
- Run Dream-7B denoising with C8 config (temp=0.5, alg_temp=0.5, top_p=1.0)
- Record intermediate partially-masked states every N denoising steps
- Label each trajectory as correct/incorrect based on answer extraction
- Save to per-shard .pt files for downstream PRM training

Usage:
    # Single GPU
    CUDA_VISIBLE_DEVICES=0 python generate_trajectories.py \
        --task gsm8k --num_trajectories 4 --shard_id 0 --num_shards 8

    # Multi-GPU launcher
    bash experiments/scripts/run_trajectory_gen.sh gsm8k 4 8

    # Merge shards after all GPUs finish
    python generate_trajectories.py --task gsm8k --merge --num_shards 8
"""

import os
import sys
import json
import argparse
import re
import time
import random
import logging
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# Project paths
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'CoDD', 'eval'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'CoDD'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

MASK_TOKEN_ID = 151666

SYSTEM_INSTRUCTIONS = {
    'gsm8k': "You are a helpful assistant.",
    'math500': (
        "You are a helpful assistant. Justify your final answer by first "
        "explaining your step-by-step derivation or reasoning. Conclude by "
        "presenting the final answer in the format: \\boxed{ANSWER}."
    ),
}

def _base_task(task):
    """Map task variants to base task for system instructions and answer verification."""
    if task.startswith('gsm8k'):
        return 'gsm8k'
    return task

DATASET_CONFIGS = {
    'gsm8k': {
        'name': 'gsm8k', 'subset': 'main', 'split': 'train',
        'question_key': 'question', 'answer_key': 'answer',
    },
    'gsm8k_test': {
        'name': 'gsm8k', 'subset': 'main', 'split': 'test',
        'question_key': 'question', 'answer_key': 'answer',
    },
    'math500': {
        'name': 'HuggingFaceH4/MATH-500', 'subset': None, 'split': 'test',
        'question_key': 'problem', 'answer_key': 'answer',
    },
}


# ---------------------------------------------------------------------------
# Trajectory recorder — hooks into Dream's denoising loop
# ---------------------------------------------------------------------------

class TrajectoryRecorder:
    """Records intermediate denoising states via generation_tokens_hook_func.

    Captures the generation region (post-prompt) at regular intervals during
    the denoising process. Each snapshot contains the token ids and the
    fraction of MASK tokens remaining.
    """

    def __init__(self, snapshot_every=6, num_steps=128):
        self.snapshot_every = snapshot_every
        self.num_steps = num_steps
        self.prompt_len = 0
        self.snapshots = []

    def reset(self, prompt_len):
        self.prompt_len = prompt_len
        self.snapshots = []

    def __call__(self, step, x, logits):
        """Hook compatible with generation_tokens_hook_func signature."""
        if step is None:
            # Initial state (before denoising loop — all MASK in gen region)
            gen = x[0, self.prompt_len:].clone().cpu().to(torch.int32)
            mr = (gen == MASK_TOKEN_ID).float().mean().item()
            self.snapshots.append((-1, gen, mr))
        elif step % self.snapshot_every == 0 or step == self.num_steps - 1:
            gen = x[0, self.prompt_len:].clone().cpu().to(torch.int32)
            mr = (gen == MASK_TOKEN_ID).float().mean().item()
            self.snapshots.append((step, gen, mr))
        return x  # return unchanged


# ---------------------------------------------------------------------------
# Answer extraction and verification
# ---------------------------------------------------------------------------

def normalize_number(s):
    """Normalize a number string for comparison."""
    s = str(s).strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        val = float(s)
        # LLaDA base model sometimes emits extremely long digit strings that
        # overflow float → inf/-inf, breaking int(val). Treat unbounded values
        # as "garbage" and fall through to the string form.
        import math
        if math.isinf(val) or math.isnan(val) or abs(val) > 1e15:
            return s.lower().strip()
        if val == int(val):
            return str(int(val))
        return f"{val:.6f}"
    except (ValueError, OverflowError):
        return s.lower().strip()


def extract_gsm8k_answer(text):
    """Extract numerical answer from GSM8K-style output.

    Tries in order: #### N, "answer is: N", \\boxed{N}, last number.
    The fallback regex requires at least one digit to avoid matching
    standalone punctuation like periods.
    """
    # Primary: GSM8K standard format
    match = re.search(r'####\s*([\-\d,\.]+)', text)
    if match:
        return normalize_number(match.group(1))
    # Secondary: "The answer is: N" or "the answer is N"
    match = re.search(r'(?:the\s+)?answer\s+is:?\s*([\-\d,\.]+)', text, re.IGNORECASE)
    if match:
        return normalize_number(match.group(1))
    # Tertiary: \boxed{N}
    match = re.search(r'\\boxed\{([\-\d,\.]+)\}', text)
    if match:
        return normalize_number(match.group(1))
    # Fallback: last number (must contain at least one digit)
    numbers = re.findall(r'-?\d[\d,]*\.?\d*', text)
    if numbers:
        return normalize_number(numbers[-1])
    return None


def extract_math_answer(text):
    """Extract answer from MATH-style output (\\boxed{...})."""
    # Find the last \\boxed{...} — handle nested braces
    depth = 0
    last_start = -1
    last_end = -1
    i = 0
    while i < len(text):
        if text[i:i+7] == '\\boxed{':
            depth = 1
            start = i + 7
            j = start
            while j < len(text) and depth > 0:
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                j += 1
            if depth == 0:
                last_start = start
                last_end = j - 1
            i = j
        else:
            i += 1
    if last_start >= 0:
        return text[last_start:last_end].strip()
    # Fallback: simpler regex
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if matches:
        return matches[-1].strip()
    return None


def extract_gsm8k_target(answer_text):
    """Extract target number from GSM8K answer field."""
    match = re.search(r'####\s*([\-\d,\.]+)', answer_text)
    if match:
        return normalize_number(match.group(1))
    return None


def verify_answer(predicted, target, task):
    """Check if predicted answer matches the ground truth."""
    if predicted is None or target is None:
        return False

    if _base_task(task) == 'gsm8k':
        return normalize_number(predicted) == normalize_number(target)
    elif _base_task(task) == 'math500':
        try:
            from math_verify import verify, parse
            gold = parse(f'\\boxed{{{target}}}')
            pred = parse(predicted)
            return verify(gold, pred)
        except Exception:
            return predicted.strip() == target.strip()
    return False


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def format_prompt(question, task, tokenizer):
    """Format a question using Dream-7B's chat template."""
    messages = [
        {'role': 'system', 'content': SYSTEM_INSTRUCTIONS[_base_task(task)]},
        {'role': 'user', 'content': question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_model(model_path):
    """Load Dream-7B model and tokenizer."""
    from dream.modeling_dream import DreamModel

    logger.info(f"Loading model from {model_path}")
    model = DreamModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model.eval()
    logger.info("Model loaded successfully")
    return model, tokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_single_trajectory(model, tokenizer, input_ids, recorder, args):
    """Run one denoising trajectory and return (generated_text, snapshots)."""
    prompt_len = input_ids.shape[1]
    recorder.reset(prompt_len)

    with torch.no_grad():
        output = model.diffusion_generate(
            input_ids,
            max_length=prompt_len + 512,
            pad_token_id=tokenizer.pad_token_id,
            steps=args.num_steps,
            temperature=args.temperature,
            top_p=args.top_p,
            alg=args.alg,
            alg_temp=args.alg_temp,
            generation_tokens_hook_func=recorder,
        )

    if hasattr(output, 'sequences'):
        output_ids = output.sequences
    else:
        output_ids = output
    if output_ids.dim() == 1:
        output_ids = output_ids.unsqueeze(0)

    # Extract generation region and truncate at EOS
    gen_ids = output_ids[0, prompt_len:]
    eos_pos = (gen_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_pos) > 0:
        gen_ids = gen_ids[:eos_pos[0]]

    # Remove remaining MASK tokens for text decoding
    gen_ids_clean = gen_ids[gen_ids != MASK_TOKEN_ID]
    text = tokenizer.decode(gen_ids_clean, skip_special_tokens=True)
    return text, recorder.snapshots


# ---------------------------------------------------------------------------
# Merge utility
# ---------------------------------------------------------------------------

def merge_shards(output_dir, num_shards, task):
    """Combine per-GPU shards into a single dataset file."""
    all_trajectories = []
    stats = {'correct': 0, 'total': 0, 'problems': set()}

    for shard_id in range(num_shards):
        path = os.path.join(output_dir, f"shard_{shard_id}.pt")
        if not os.path.exists(path):
            logger.warning(f"Shard {shard_id} not found: {path}")
            continue
        data = torch.load(path, weights_only=False)
        all_trajectories.extend(data)
        for t in data:
            stats['total'] += 1
            if t['is_correct']:
                stats['correct'] += 1
            stats['problems'].add(t['problem_id'])
        logger.info(f"Loaded shard {shard_id}: {len(data)} trajectories")

    merged_path = os.path.join(output_dir, "trajectories_all.pt")
    torch.save(all_trajectories, merged_path)

    acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    summary = {
        'task': task,
        'total_trajectories': stats['total'],
        'total_problems': len(stats['problems']),
        'correct': stats['correct'],
        'accuracy': acc,
        'merged_from_shards': num_shards,
    }
    summary_path = os.path.join(output_dir, "summary_merged.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Merged {stats['total']} trajectories from {len(stats['problems'])} problems. "
        f"Correct rate: {acc:.1%} ({stats['correct']}/{stats['total']}). "
        f"Saved to {merged_path}"
    )
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate denoising trajectories for PRM training"
    )
    parser.add_argument("--task", type=str, required=True,
                        choices=["gsm8k", "gsm8k_test", "math500"])
    parser.add_argument("--model_path", type=str,
                        default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_trajectories", type=int, default=4,
                        help="Trajectories per problem (different seeds)")
    parser.add_argument("--num_steps", type=int, default=128,
                        help="Denoising steps")
    parser.add_argument("--snapshot_every", type=int, default=6,
                        help="Record snapshot every N denoising steps")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Token sampling temperature (C8 config)")
    parser.add_argument("--alg_temp", type=float, default=0.5,
                        help="Unmask-order temperature (C8 config)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p for sampling (C8 config)")
    parser.add_argument("--alg", type=str, default="entropy",
                        help="Unmask selection algorithm")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=50,
                        help="Save checkpoint every N problems")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all shards instead of generating")
    args = parser.parse_args()

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            PROJECT_ROOT, 'data', 'prm_trajectories', args.task
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Merge mode
    if args.merge:
        merge_shards(args.output_dir, args.num_shards, args.task)
        return

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    cfg = DATASET_CONFIGS[args.task]
    if cfg['subset']:
        dataset = load_dataset(cfg['name'], cfg['subset'], split=cfg['split'])
    else:
        dataset = load_dataset(cfg['name'], split=cfg['split'])

    total_problems = len(dataset)
    logger.info(f"Dataset {args.task}: {total_problems} samples")

    # Shard (interleaved for load balance)
    shard_indices = list(range(args.shard_id, total_problems, args.num_shards))
    logger.info(
        f"Shard {args.shard_id}/{args.num_shards}: "
        f"{len(shard_indices)} problems to process"
    )

    # -----------------------------------------------------------------------
    # Resume support
    # -----------------------------------------------------------------------
    progress_file = os.path.join(
        args.output_dir, f"progress_shard{args.shard_id}.json"
    )
    output_file = os.path.join(args.output_dir, f"shard_{args.shard_id}.pt")

    completed_indices = set()
    trajectories = []
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            progress = json.load(f)
        completed_indices = set(progress.get("completed", []))
        logger.info(f"Resuming: {len(completed_indices)} problems already done")
    if os.path.exists(output_file):
        trajectories = torch.load(output_file, weights_only=False)
        logger.info(f"Loaded {len(trajectories)} existing trajectories")

    remaining = [i for i in shard_indices if i not in completed_indices]
    if not remaining:
        logger.info("All problems in this shard are complete!")
        return

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    model, tokenizer = load_model(args.model_path)
    recorder = TrajectoryRecorder(
        snapshot_every=args.snapshot_every,
        num_steps=args.num_steps,
    )

    # -----------------------------------------------------------------------
    # Generate
    # -----------------------------------------------------------------------
    start_time = time.time()
    problems_done_this_run = 0

    for local_idx, problem_idx in enumerate(remaining):
        sample = dataset[problem_idx]
        question = sample[cfg['question_key']]

        # Ground truth
        if _base_task(args.task) == 'gsm8k':
            target = extract_gsm8k_target(sample[cfg['answer_key']])
        else:  # math500
            target = sample[cfg['answer_key']]

        # Format and tokenize prompt
        prompt_text = format_prompt(question, args.task, tokenizer)
        input_ids = tokenizer(
            prompt_text, return_tensors='pt'
        )['input_ids'].cuda()
        prompt_len = input_ids.shape[1]

        for traj_idx in range(args.num_trajectories):
            seed = args.base_seed + problem_idx * 100 + traj_idx
            set_seed(seed)

            try:
                text, snapshots = generate_single_trajectory(
                    model, tokenizer, input_ids, recorder, args
                )
            except Exception as e:
                logger.error(
                    f"Error on problem {problem_idx} traj {traj_idx}: {e}"
                )
                continue

            # Extract and verify answer
            if _base_task(args.task) == 'gsm8k':
                predicted = extract_gsm8k_answer(text)
            else:
                predicted = extract_math_answer(text)
            is_correct = verify_answer(predicted, target, args.task)

            # Pack trajectory data
            snapshot_steps = [s[0] for s in snapshots]
            gen_snapshots = torch.stack([s[1] for s in snapshots])
            mask_ratios = [s[2] for s in snapshots]

            trajectory = {
                'problem_id': problem_idx,
                'trajectory_id': traj_idx,
                'seed': seed,
                'task': args.task,
                'prompt_ids': input_ids[0].cpu().to(torch.int32),
                'gen_snapshots': gen_snapshots,   # [N_snap, gen_len]
                'snapshot_steps': snapshot_steps,
                'mask_ratios': mask_ratios,
                'final_text': text,
                'answer_extracted': str(predicted),
                'answer_target': str(target),
                'is_correct': is_correct,
            }
            trajectories.append(trajectory)

        completed_indices.add(problem_idx)
        problems_done_this_run += 1

        # Progress logging
        elapsed = time.time() - start_time
        if problems_done_this_run > 0:
            per_problem = elapsed / problems_done_this_run
            eta_h = per_problem * (len(remaining) - problems_done_this_run) / 3600
        else:
            eta_h = 0

        if problems_done_this_run % 10 == 0 or problems_done_this_run == 1:
            n_correct = sum(1 for t in trajectories if t['is_correct'])
            n_total = len(trajectories)
            logger.info(
                f"[{len(completed_indices)}/{len(shard_indices)}] "
                f"problem={problem_idx} "
                f"correct={n_correct}/{n_total} ({n_correct/n_total:.1%}) "
                f"speed={per_problem:.1f}s/problem "
                f"ETA={eta_h:.1f}h"
            )

        # Periodic save
        if problems_done_this_run % args.save_every == 0:
            torch.save(trajectories, output_file)
            with open(progress_file, 'w') as f:
                json.dump({"completed": sorted(completed_indices)}, f)
            logger.info(f"Checkpoint: {len(trajectories)} trajectories saved")

    # -----------------------------------------------------------------------
    # Final save
    # -----------------------------------------------------------------------
    torch.save(trajectories, output_file)
    with open(progress_file, 'w') as f:
        json.dump({"completed": sorted(completed_indices)}, f)

    n_correct = sum(1 for t in trajectories if t['is_correct'])
    n_total = len(trajectories)
    logger.info(
        f"Shard {args.shard_id} complete! "
        f"{n_total} trajectories from {len(completed_indices)} problems. "
        f"Correct: {n_correct}/{n_total} ({n_correct/n_total:.1%})"
    )

    # Save summary
    summary = {
        'task': args.task,
        'shard_id': args.shard_id,
        'num_shards': args.num_shards,
        'total_trajectories': n_total,
        'correct': n_correct,
        'accuracy': n_correct / n_total if n_total > 0 else 0,
        'num_problems': len(completed_indices),
        'config': {
            'temperature': args.temperature,
            'alg_temp': args.alg_temp,
            'top_p': args.top_p,
            'alg': args.alg,
            'num_steps': args.num_steps,
            'snapshot_every': args.snapshot_every,
            'num_trajectories': args.num_trajectories,
            'base_seed': args.base_seed,
        },
        'wall_time_seconds': time.time() - start_time,
    }
    summary_file = os.path.join(
        args.output_dir, f"summary_shard{args.shard_id}.json"
    )
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
