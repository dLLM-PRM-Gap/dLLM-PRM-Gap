"""
Phase 3 Eval: PRM Step-level Accuracy by Mask Ratio.

Validates Claim C4: PRM prediction accuracy monotonically improves
as mask_ratio decreases (more tokens revealed → easier to judge).

Also compares DiffusionPRM vs ORM on partial states to support Claim C3.

Usage:
    # Evaluate DiffusionPRM
    CUDA_VISIBLE_DEVICES=0 python eval_prm_step_accuracy.py \
        --trajectory_dir data/prm_trajectories/gsm8k \
        --prm_checkpoint checkpoints/prm_gsm8k/best.pt \
        --output_dir eval_results/prm_step_accuracy

    # Compare DiffusionPRM vs ORM
    CUDA_VISIBLE_DEVICES=0 python eval_prm_step_accuracy.py \
        --trajectory_dir data/prm_trajectories/gsm8k \
        --prm_checkpoint checkpoints/prm_gsm8k/best.pt \
        --orm_checkpoint checkpoints/orm_gsm8k/best.pt \
        --output_dir eval_results/prm_step_accuracy
"""

import os
import sys
import json
import argparse
import logging
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'CoDD', 'eval'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'CoDD'))

from prm.model import DiffusionPRM, OutcomeRewardModel
from prm.dataset import DiffusionPRMDataset, collate_fn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)

BUCKET_SIZE = 0.1
NUM_BUCKETS = 10


def load_prm(checkpoint_path, model_path, device, config=None):
    """Load trained DiffusionPRM from checkpoint."""
    from dream.modeling_dream import DreamModel

    logger.info(f"Loading PRM backbone from {model_path}")
    backbone = DreamModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )

    # Load checkpoint to CPU first (much faster — avoids loading optimizer states to GPU)
    logger.info(f"Loading PRM checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    ckpt_config = ckpt.get('config', {})

    model = DiffusionPRM(
        backbone,
        hidden_size=backbone.config.hidden_size,
        step_embed_dim=ckpt_config.get('step_embed_dim', 256),
        reward_hidden=ckpt_config.get('reward_hidden', 1024),
        lora_r=ckpt_config.get('lora_r', 16),
        lora_alpha=ckpt_config.get('lora_alpha', 32),
        lora_dropout=ckpt_config.get('lora_dropout', 0.05),
        causal=ckpt_config.get('causal', False),
        no_step_embed=ckpt_config.get('no_step_embed', False),
        no_mask_aware=ckpt_config.get('no_mask_aware', False),
    )

    state_dict = ckpt.get('model_state_dict', ckpt)
    del ckpt  # Free the full checkpoint (optimizer states etc.) immediately
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    # Move non-backbone modules to device (backbone already on CUDA via device_map)
    if not model.no_step_embed:
        model.step_embed.to(device)
        model.step_proj.to(device)
    model.reward_head.to(device)
    model.eval()
    logger.info(f"DiffusionPRM loaded from {checkpoint_path}")
    return model


def load_orm(checkpoint_path, model_path, device):
    """Load trained ORM from checkpoint."""
    from dream.modeling_dream import DreamModel

    logger.info(f"Loading ORM backbone from {model_path}")
    backbone = DreamModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )

    logger.info(f"Loading ORM checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    ckpt_config = ckpt.get('config', {})

    model = OutcomeRewardModel(
        backbone,
        hidden_size=backbone.config.hidden_size,
        lora_r=ckpt_config.get('lora_r', 16),
        lora_alpha=ckpt_config.get('lora_alpha', 32),
        lora_dropout=ckpt_config.get('lora_dropout', 0.05),
    )

    state_dict = ckpt.get('model_state_dict', ckpt)
    del ckpt
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    # Move non-backbone modules to device (backbone already on CUDA via device_map)
    model.reward_head.to(device)
    model.eval()
    logger.info(f"ORM loaded from {checkpoint_path}")
    return model


def bucket_key(mask_ratio):
    """Map mask_ratio to bucket string like '0.3-0.4'."""
    b = min(int(mask_ratio / BUCKET_SIZE), NUM_BUCKETS - 1)
    return f"{b * BUCKET_SIZE:.1f}-{(b + 1) * BUCKET_SIZE:.1f}"


def evaluate_prm_by_bucket(model, dataset, device, batch_size=8):
    """Evaluate DiffusionPRM accuracy by mask_ratio bucket.

    Returns:
        bucket_metrics: dict mapping bucket_key → {accuracy, loss, n_samples,
                                                    n_correct, precision, recall}
        overall: dict with overall accuracy, loss
    """
    from torch.utils.data import DataLoader
    from functools import partial

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=partial(collate_fn), num_workers=2, pin_memory=True,
    )

    # Per-bucket tracking
    buckets = defaultdict(lambda: {
        'correct': 0, 'total': 0, 'loss_sum': 0.0,
        'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
    })
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            prompt_lens = batch['prompt_lens'].to(device)
            mask_ratios = batch['mask_ratios'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, prompt_lens, mask_ratios, attention_mask=attention_mask)
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, reduction='none'
            )
            preds = (logits > 0).float()

            for mr, pred, label, l in zip(
                mask_ratios.cpu(), preds.cpu(), labels.cpu(), loss.cpu()
            ):
                key = bucket_key(mr.item())
                b = buckets[key]
                b['total'] += 1
                b['loss_sum'] += l.item()
                is_correct = int(pred == label)
                b['correct'] += is_correct

                # Confusion matrix
                if pred == 1 and label == 1:
                    b['tp'] += 1
                elif pred == 1 and label == 0:
                    b['fp'] += 1
                elif pred == 0 and label == 1:
                    b['fn'] += 1
                else:
                    b['tn'] += 1

                total_correct += is_correct
                total_samples += 1
                total_loss += l.item()

    # Compute per-bucket metrics
    bucket_metrics = {}
    for key in sorted(buckets.keys()):
        b = buckets[key]
        n = b['total']
        bucket_metrics[key] = {
            'accuracy': b['correct'] / n if n > 0 else 0,
            'loss': b['loss_sum'] / n if n > 0 else 0,
            'n_samples': n,
            'n_correct': b['correct'],
            'precision': b['tp'] / (b['tp'] + b['fp']) if (b['tp'] + b['fp']) > 0 else 0,
            'recall': b['tp'] / (b['tp'] + b['fn']) if (b['tp'] + b['fn']) > 0 else 0,
        }

    overall = {
        'accuracy': total_correct / max(total_samples, 1),
        'loss': total_loss / max(total_samples, 1),
        'n_samples': total_samples,
    }

    return bucket_metrics, overall


def evaluate_orm_on_partial(model, dataset, device, batch_size=8):
    """Evaluate ORM on partial states (it wasn't trained on these).

    ORM only sees input_ids + prompt_lens, ignoring mask_ratios.
    This tests whether ORM generalizes to partial denoising states.
    """
    from torch.utils.data import DataLoader
    from functools import partial

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=partial(collate_fn), num_workers=2, pin_memory=True,
    )

    buckets = defaultdict(lambda: {'correct': 0, 'total': 0})
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            prompt_lens = batch['prompt_lens'].to(device)
            mask_ratios = batch['mask_ratios']
            labels = batch['labels'].to(device)

            # ORM forward: no mask_ratios input
            logits = model(input_ids, prompt_lens, attention_mask=attention_mask)
            preds = (logits > 0).float()

            for mr, pred, label in zip(
                mask_ratios.cpu(), preds.cpu(), labels.cpu()
            ):
                key = bucket_key(mr.item())
                is_correct = int(pred == label)
                buckets[key]['total'] += 1
                buckets[key]['correct'] += is_correct
                total_correct += is_correct
                total_samples += 1

    bucket_metrics = {}
    for key in sorted(buckets.keys()):
        b = buckets[key]
        n = b['total']
        bucket_metrics[key] = {
            'accuracy': b['correct'] / n if n > 0 else 0,
            'n_samples': n,
        }

    overall = {
        'accuracy': total_correct / max(total_samples, 1),
        'n_samples': total_samples,
    }

    return bucket_metrics, overall


def check_monotonicity(bucket_metrics):
    """Check if accuracy monotonically increases as mask_ratio decreases.

    Sorted order is ascending mask_ratio: [0.0-0.1, 0.1-0.2, ..., 0.9-1.0]
    Claim C4 says accuracy should DECREASE along this order (more mask → harder).
    A violation = accuracy goes UP when moving to higher mask_ratio bucket.

    Returns:
        is_monotonic: bool
        violations: list of tuples
        trend_score: Spearman correlation between mask_ratio and accuracy
    """
    from scipy.stats import spearmanr

    sorted_buckets = sorted(bucket_metrics.items())
    # sorted_buckets[0] = lowest mask_ratio (0.0-0.1) → highest expected accuracy
    # sorted_buckets[-1] = highest mask_ratio (0.9-1.0) → lowest expected accuracy

    accs = [v['accuracy'] for _, v in sorted_buckets]
    n_samples = [v['n_samples'] for _, v in sorted_buckets]

    violations = []
    for i in range(len(sorted_buckets) - 1):
        key_lo_mr = sorted_buckets[i][0]      # lower mask_ratio bucket
        key_hi_mr = sorted_buckets[i + 1][0]  # higher mask_ratio bucket
        acc_lo_mr = sorted_buckets[i][1]['accuracy']   # should be HIGHER
        acc_hi_mr = sorted_buckets[i + 1][1]['accuracy']  # should be LOWER
        # Violation: higher mask_ratio has HIGHER accuracy (wrong direction)
        if acc_hi_mr > acc_lo_mr + 0.01:  # 1% tolerance
            violations.append((key_lo_mr, key_hi_mr, acc_lo_mr, acc_hi_mr))

    # Spearman: mask_ratio midpoints vs accuracy
    midpoints = [(i + 0.5) * BUCKET_SIZE for i in range(len(accs))]
    if len(accs) >= 3:
        corr, pval = spearmanr(midpoints, accs)
    else:
        corr, pval = 0.0, 1.0

    return {
        'is_monotonic': len(violations) == 0,
        'violations': violations,
        'spearman_correlation': corr,  # negative = accuracy goes up as mask_ratio goes down (desired)
        'spearman_pvalue': pval,
        'bucket_order': [k for k, _ in sorted_buckets],
        'accuracies': accs,
        'sample_counts': n_samples,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PRM step-level accuracy by mask ratio bucket"
    )
    parser.add_argument("--trajectory_dir", type=str, required=True)
    parser.add_argument("--prm_checkpoint", type=str, required=True,
                        help="DiffusionPRM checkpoint path")
    parser.add_argument("--orm_checkpoint", type=str, default=None,
                        help="ORM checkpoint for comparison (optional)")
    parser.add_argument("--model_path", type=str,
                        default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--output_dir", type=str,
                        default="eval_results/prm_step_accuracy")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")

    # ------------------------------------------------------------------
    # Load trajectory data
    # ------------------------------------------------------------------
    import random
    import glob

    merged = os.path.join(args.trajectory_dir, "trajectories_all.pt")
    if os.path.exists(merged):
        trajectories = torch.load(merged, weights_only=False)
    else:
        shard_files = sorted(glob.glob(
            os.path.join(args.trajectory_dir, "shard_*.pt")
        ))
        trajectories = []
        for f in shard_files:
            trajectories.extend(torch.load(f, weights_only=False))
    logger.info(f"Loaded {len(trajectories)} trajectories")

    # Use same val split as training
    rng = random.Random(args.seed)
    problem_ids = sorted(set(t['problem_id'] for t in trajectories))
    rng.shuffle(problem_ids)
    n_val = max(50, int(len(problem_ids) * 0.1))
    val_pids = set(problem_ids[:n_val])
    val_trajs = [t for t in trajectories if t['problem_id'] in val_pids]
    logger.info(f"Val set: {len(val_trajs)} trajectories, {n_val} problems")

    # Build dataset with ALL mask ratios
    val_dataset = DiffusionPRMDataset(
        val_trajs, min_mask_ratio=0.0, max_mask_ratio=1.0
    )
    logger.info(f"Val samples (all steps): {len(val_dataset)}")

    # ------------------------------------------------------------------
    # Evaluate DiffusionPRM
    # ------------------------------------------------------------------
    prm_model = load_prm(args.prm_checkpoint, args.model_path, device)
    prm_buckets, prm_overall = evaluate_prm_by_bucket(
        prm_model, val_dataset, device, args.batch_size
    )

    logger.info("=" * 60)
    logger.info("DiffusionPRM Step-level Accuracy by Mask Ratio")
    logger.info("=" * 60)
    logger.info(f"Overall: acc={prm_overall['accuracy']:.1%} "
                f"loss={prm_overall['loss']:.4f} "
                f"n={prm_overall['n_samples']}")
    logger.info("-" * 60)
    for key, metrics in sorted(prm_buckets.items()):
        logger.info(
            f"  mask_ratio {key}: "
            f"acc={metrics['accuracy']:.1%} "
            f"prec={metrics['precision']:.1%} "
            f"recall={metrics['recall']:.1%} "
            f"n={metrics['n_samples']}"
        )

    # Check monotonicity (Claim C4)
    mono = check_monotonicity(prm_buckets)
    logger.info("-" * 60)
    logger.info(f"Monotonicity: {'PASS' if mono['is_monotonic'] else 'FAIL'}")
    logger.info(f"Spearman(mask_ratio, accuracy): r={mono['spearman_correlation']:.3f} "
                f"p={mono['spearman_pvalue']:.4f}")
    if mono['violations']:
        logger.info("Violations:")
        for v in mono['violations']:
            logger.info(f"  {v[0]} ({v[2]:.1%}) > {v[1]} ({v[3]:.1%})")

    results = {
        'prm': {
            'overall': prm_overall,
            'buckets': prm_buckets,
            'monotonicity': {
                'is_monotonic': mono['is_monotonic'],
                'spearman_correlation': mono['spearman_correlation'],
                'spearman_pvalue': mono['spearman_pvalue'],
                'violations': [
                    {'high': v[0], 'low': v[1],
                     'acc_high': v[2], 'acc_low': v[3]}
                    for v in mono['violations']
                ],
            },
        }
    }

    # Free PRM model memory before loading ORM
    del prm_model
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Evaluate ORM on partial states (optional comparison)
    # ------------------------------------------------------------------
    if args.orm_checkpoint:
        orm_model = load_orm(args.orm_checkpoint, args.model_path, device)
        orm_buckets, orm_overall = evaluate_orm_on_partial(
            orm_model, val_dataset, device, args.batch_size
        )

        logger.info("=" * 60)
        logger.info("ORM on Partial States (not trained on these)")
        logger.info("=" * 60)
        logger.info(f"Overall: acc={orm_overall['accuracy']:.1%} "
                    f"n={orm_overall['n_samples']}")
        for key, metrics in sorted(orm_buckets.items()):
            logger.info(
                f"  mask_ratio {key}: "
                f"acc={metrics['accuracy']:.1%} "
                f"n={metrics['n_samples']}"
            )

        results['orm'] = {
            'overall': orm_overall,
            'buckets': orm_buckets,
        }

        # PRM vs ORM comparison per bucket
        logger.info("=" * 60)
        logger.info("PRM vs ORM per Bucket")
        logger.info("=" * 60)
        comparison = {}
        for key in sorted(prm_buckets.keys()):
            prm_acc = prm_buckets[key]['accuracy']
            orm_acc = orm_buckets.get(key, {}).get('accuracy', 0)
            delta = prm_acc - orm_acc
            comparison[key] = {
                'prm_acc': prm_acc,
                'orm_acc': orm_acc,
                'delta': delta,
            }
            logger.info(
                f"  {key}: PRM={prm_acc:.1%} ORM={orm_acc:.1%} "
                f"Δ={delta:+.1%}"
            )
        results['comparison'] = comparison

        del orm_model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    task_name = os.path.basename(args.trajectory_dir)
    out_path = os.path.join(
        args.output_dir, f"prm_step_accuracy_{task_name}.json"
    )
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # Claim C4 verdict
    logger.info("=" * 60)
    if mono['is_monotonic']:
        logger.info("CLAIM C4 SUPPORTED: PRM accuracy monotonically improves "
                    "as mask_ratio decreases")
    else:
        logger.info("CLAIM C4 PARTIALLY SUPPORTED: monotonicity has "
                    f"{len(mono['violations'])} violation(s)")
    logger.info(f"Spearman correlation: {mono['spearman_correlation']:.3f} "
                f"(expect negative; p={mono['spearman_pvalue']:.4f})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
