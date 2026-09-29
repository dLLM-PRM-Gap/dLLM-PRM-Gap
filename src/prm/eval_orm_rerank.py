"""
Phase 2 Eval: ORM-Rerank@N on Best-of-N solutions.

For each test problem, generate N solutions, score with ORM, select highest.
Compares against Majority@N and Oracle@N from Best-of-N diagnostic.

Usage:
    # Generate N solutions + rerank (from scratch)
    CUDA_VISIBLE_DEVICES=0 python eval_orm_rerank.py \
        --task gsm8k --num_solutions 6 \
        --orm_checkpoint checkpoints/orm_gsm8k/best.pt

    # Rerank using existing trajectory data
    CUDA_VISIBLE_DEVICES=0 python eval_orm_rerank.py \
        --task gsm8k --from_trajectories data/prm_trajectories/gsm8k/trajectories_all.pt \
        --orm_checkpoint checkpoints/orm_gsm8k/best.pt
"""

import os
import sys
import json
import argparse
import re
import logging
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'CoDD', 'eval'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'CoDD'))

from prm.model import OutcomeRewardModel, MASK_TOKEN_ID
from prm.dataset import collate_fn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def load_orm(checkpoint_path, model_path, device):
    """Load trained ORM from checkpoint.

    Matches training procedure: load backbone on CPU → LoRA wrap → move to GPU.
    Using device_map="cuda" with PEFT can cause issues with dispatched models.
    """
    from dream.modeling_dream import DreamModel

    logger.info(f"Loading backbone from {model_path}")
    backbone = DreamModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    logger.info("Creating ORM with LoRA (on CPU)...")
    model = OutcomeRewardModel(
        backbone,
        hidden_size=backbone.config.hidden_size,
    )

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)

    logger.info("Moving model to GPU...")
    model = model.to(device)
    model.eval()
    logger.info(f"ORM loaded from {checkpoint_path} (val_acc={ckpt.get('best_val_acc', 'N/A')})")
    return model


def score_solutions(model, solutions, device, batch_size=8):
    """Score a list of solutions with ORM.

    Each solution: dict with 'input_ids' (tensor) and 'prompt_len' (int).
    Returns: list of float scores.
    """
    scores = []
    for i in range(0, len(solutions), batch_size):
        batch = solutions[i:i + batch_size]
        collated = collate_fn(batch)
        input_ids = collated['input_ids'].to(device)
        attention_mask = collated['attention_mask'].to(device)
        prompt_lens = collated['prompt_lens'].to(device)

        with torch.no_grad():
            logits = model(input_ids, prompt_lens, attention_mask=attention_mask)
        scores.extend(logits.cpu().tolist())
    return scores


def rerank_from_trajectories(trajectory_path, model, device, val_only=False, val_seed=42):
    """Rerank using pre-generated trajectory data.

    Groups trajectories by problem_id, scores final outputs with ORM,
    selects highest-scored per problem.

    If val_only=True, use only the validation split (same logic as training)
    to avoid data leakage from training set.
    """
    import random as _random
    trajectories = torch.load(trajectory_path, weights_only=False)
    logger.info(f"Loaded {len(trajectories)} trajectories")

    if val_only:
        rng = _random.Random(val_seed)
        problem_ids = sorted(set(t['problem_id'] for t in trajectories))
        rng.shuffle(problem_ids)
        n_val = max(50, int(len(problem_ids) * 0.1))
        val_pids = set(problem_ids[:n_val])
        trajectories = [t for t in trajectories if t['problem_id'] in val_pids]
        logger.info(f"Val-only: {len(trajectories)} trajectories from {n_val} problems")

    # Group by problem
    by_problem = defaultdict(list)
    for traj in trajectories:
        by_problem[traj['problem_id']].append(traj)

    results = {}
    all_solutions = []
    solution_map = []  # (problem_id, traj_idx, is_correct)

    for pid in sorted(by_problem.keys()):
        trajs = by_problem[pid]
        for traj in trajs:
            final_gen = traj['gen_snapshots'][-1]
            prompt = traj['prompt_ids']
            all_solutions.append({
                'input_ids': torch.cat([prompt.long(), final_gen.long()]),
                'prompt_len': len(prompt),
                'label': 0.0,  # dummy label for collate_fn
            })
            solution_map.append((pid, traj['trajectory_id'], traj['is_correct']))

    logger.info(f"Scoring {len(all_solutions)} solutions across {len(by_problem)} problems")
    scores = score_solutions(model, all_solutions, device)

    # Group scores by problem and pick best
    problem_scores = defaultdict(list)
    for (pid, tid, correct), score in zip(solution_map, scores):
        problem_scores[pid].append((score, correct))

    orm_correct = 0
    majority_correct = 0
    oracle_correct = 0
    total = len(problem_scores)

    for pid, items in problem_scores.items():
        # ORM-Rerank: pick highest score
        best_idx = max(range(len(items)), key=lambda i: items[i][0])
        orm_correct += int(items[best_idx][1])

        # Majority voting
        n_correct = sum(1 for _, c in items if c)
        majority_correct += int(n_correct > len(items) / 2)

        # Oracle: any correct
        oracle_correct += int(any(c for _, c in items))

    N = len(next(iter(problem_scores.values())))
    results = {
        'task': os.path.basename(os.path.dirname(trajectory_path)),
        'num_problems': total,
        'num_solutions_per_problem': N,
        'orm_rerank_accuracy': orm_correct / total,
        'majority_accuracy': majority_correct / total,
        'oracle_accuracy': oracle_correct / total,
        'orm_correct': orm_correct,
        'majority_correct': majority_correct,
        'oracle_correct': oracle_correct,
    }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="ORM-Rerank evaluation on Best-of-N"
    )
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--orm_checkpoint", type=str, required=True)
    parser.add_argument("--model_path", type=str,
                        default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--from_trajectories", type=str, default=None,
                        help="Path to trajectory .pt file for reranking")
    parser.add_argument("--output_dir", type=str,
                        default="eval_results/orm_rerank")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_only", action="store_true",
                        help="Evaluate on val split only (same split as training)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")

    model = load_orm(args.orm_checkpoint, args.model_path, device)

    if args.from_trajectories:
        results = rerank_from_trajectories(
            args.from_trajectories, model, device,
            val_only=args.val_only,
        )
    else:
        raise NotImplementedError(
            "Online generation not yet implemented. "
            "Use --from_trajectories with pre-generated data."
        )

    # Print results
    logger.info("=" * 60)
    logger.info(f"Task: {results.get('task', args.task)}")
    logger.info(f"Problems: {results['num_problems']}")
    logger.info(f"Solutions/problem: {results['num_solutions_per_problem']}")
    logger.info(f"ORM-Rerank:  {results['orm_rerank_accuracy']:.1%}")
    logger.info(f"Majority:    {results['majority_accuracy']:.1%}")
    logger.info(f"Oracle:      {results['oracle_accuracy']:.1%}")
    logger.info("=" * 60)

    # Save
    out_path = os.path.join(
        args.output_dir,
        f"orm_rerank_{args.task}.json"
    )
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
