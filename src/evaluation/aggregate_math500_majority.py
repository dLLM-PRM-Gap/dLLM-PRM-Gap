#!/usr/bin/env python3
"""Canonical-clustering Majority@N for MATH500 reasoning tasks.

The naive majority voting on string-extracted answers fails on MATH500 because
LaTeX answers like "1/2", "\\frac{1}{2}", and "0.5" are mathematically equivalent
but textually distinct. This script uses the LaTeX-aware ``math_verify`` parser
to reduce each trajectory's ``\\boxed{...}`` answer to a canonical SymPy form,
clusters trajectories by canonical equivalence, and selects the largest cluster
as the majority answer.

This is the script that produces the MATH500 Majority@N numbers reported in the
paper appendix (Majority@32 = 17.20% on MATH500 with 2-seed averaging).

Inputs
------
``data/prm_trajectories/math500_vanilla_N32{,_s43,_s44,...}/shard_*.pt``
    Trajectory shards saved by ``generate_trajectories.py`` containing
    ``final_text`` (raw LaTeX solution) and ``answer_target`` (gold answer).

Outputs
-------
``eval_results/math500_majority_aggregated.json``
    Per-seed and pooled Majority@N accuracies across N in {2, 4, 8, 16, 32}.

Usage
-----
    python -m src.evaluation.aggregate_math500_majority \\
        --trajectory_dir data/prm_trajectories \\
        --output_path eval_results/math500_majority_aggregated.json \\
        --seed_tags 42,43

Dependencies
------------
``math_verify`` (pip install math-verify)
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import torch  # noqa: F401  (torch is needed for `torch.load` of legacy .pt files)
from math_verify.parser import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    parse,
)

from .math_verify_util import compute_score


def canonical_repr(text: str) -> str | None:
    """Parse free-form text into a canonical SymPy string.

    Returns ``None`` if no parseable expression is found.
    """
    if not text:
        return None
    try:
        res = parse(
            text,
            extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
        )
        if res:
            return str(res[0])
    except Exception:
        pass
    return None


def aggregate_one_seed(trajectory_dir: str, seed_tag: str) -> dict:
    suffix = "" if seed_tag == "42" else f"_s{seed_tag}"
    pattern = os.path.join(
        trajectory_dir, f"math500_vanilla_N32{suffix}", "shard_*.pt"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        return {}

    by_prob = defaultdict(list)
    for f in files:
        for traj in torch.load(f, weights_only=False):
            by_prob[traj["problem_id"]].append(traj)

    parsed = {}
    for pid, trajs in by_prob.items():
        trajs_sorted = sorted(trajs, key=lambda t: t.get("trajectory_id", 0))
        for i, t in enumerate(trajs_sorted):
            parsed[(pid, i)] = canonical_repr(t.get("final_text", ""))

    seed_result = {}
    for N in (2, 4, 8, 16, 32):
        n_correct = 0
        n_total = 0
        for pid, trajs in by_prob.items():
            trajs_sorted = sorted(
                trajs, key=lambda t: t.get("trajectory_id", 0)
            )[:N]
            if len(trajs_sorted) < N:
                continue
            canons = [parsed.get((pid, i)) for i in range(N)]
            valid = [c for c in canons if c is not None]
            if not valid:
                n_total += 1
                continue
            cnt = Counter(valid)
            maj_canon, _ = cnt.most_common(1)[0]
            rep_traj = next(
                trajs_sorted[i]
                for i, c in enumerate(canons)
                if c == maj_canon
            )
            gold = trajs_sorted[0].get("answer_target", "")
            try:
                if compute_score(rep_traj["final_text"], gold) > 0.5:
                    n_correct += 1
            except Exception:
                pass
            n_total += 1
        acc = n_correct / n_total * 100 if n_total else 0
        seed_result[f"N={N}"] = {
            "acc": round(acc, 2),
            "n_correct": n_correct,
            "n_total": n_total,
        }
    return seed_result


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--trajectory_dir", default="data/prm_trajectories")
    ap.add_argument(
        "--output_path",
        default="eval_results/math500_majority_aggregated.json",
    )
    ap.add_argument(
        "--seed_tags",
        default="42,43",
        help="Comma-separated seed tags whose trajectory shards live under "
        "math500_vanilla_N32{,_sXX} subdirectories.",
    )
    args = ap.parse_args()

    results = {}
    seeds = args.seed_tags.split(",")
    for seed in seeds:
        seed = seed.strip()
        seed_res = aggregate_one_seed(args.trajectory_dir, seed)
        if seed_res:
            results[f"s{seed}"] = seed_res

    print("\n=== MATH500 Majority@N (math-verify canonical, multi-seed) ===")
    for N in (2, 4, 8, 16, 32):
        accs = [results[k][f"N={N}"]["acc"] for k in results]
        if accs:
            mean = sum(accs) / len(accs)
            print(
                f"  Majority@{N}: {mean:.2f}% (per-seed: {accs}, n={len(accs)})"
            )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output_path}")


if __name__ == "__main__":
    main()
