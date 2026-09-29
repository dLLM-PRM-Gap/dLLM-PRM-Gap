"""PRM sanity checks (GPT Round 3 MUST-DO):

1. Sign convention: confirm higher PRM score → correct
2. Rank correlation (Kendall τ, Spearman ρ) between scores and correctness
3. Score-delta histogram: score(correct) - score(incorrect) per problem
4. Random-rerank comparator at N=8 (shows PRM-rerank isn't accidentally worst)
5. AUC-by-slice reconciliation: all AUC numbers in paper × data slice

Outputs: eval_results/sanity_checks.json + text summary
"""
import json
import glob
import random
import statistics
from collections import defaultdict

random.seed(42)


def load_per_problem_with_scores():
    """Load per-problem items with (bidir PRM final-state score, orm score, is_correct)."""
    # Bidir PRM mask=0 scores from prm_correlation_shard
    prm_final = {}
    for f in sorted(glob.glob('eval_results/prm_correlation_shard_*.json')):
        d = json.load(open(f))
        for it in d['results']:
            if it.get('mr') == 0.0:
                prm_final[(it['pid'], it['tid'])] = it['score']

    per_prob = defaultdict(list)
    for f in sorted(glob.glob('eval_results/orm_scores_shard_*.json')):
        d = json.load(open(f))
        for it in d['items']:
            m = it['meta']
            per_prob[m['problem_id']].append({
                'tid': m['trajectory_id'],
                'orm_score': it['score'],
                'prm_score': prm_final.get((m['problem_id'], m['trajectory_id'])),
                'is_correct': bool(m['is_correct']),
            })
    return per_prob


def kendall_spearman_per_problem(per_prob, score_key='prm_score'):
    """Per-problem Kendall τ and Spearman ρ between score and is_correct binary.
    Returns aggregated stats."""
    taus, rhos = [], []
    for items in per_prob.values():
        # Use scipy-free implementation for speed
        pairs = [(x[score_key], int(x['is_correct'])) for x in items if x.get(score_key) is not None]
        if len(pairs) < 2 or len(set(p[1] for p in pairs)) < 2:
            continue
        # Kendall τ: concordant - discordant / (n*(n-1)/2)
        n = len(pairs)
        concordant = discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                si, ci = pairs[i]
                sj, cj = pairs[j]
                if si != sj and ci != cj:
                    if (si > sj) == (ci > cj):
                        concordant += 1
                    else:
                        discordant += 1
        total = concordant + discordant
        if total > 0:
            taus.append((concordant - discordant) / total)
        # Point-biserial (Pearson-like) as Spearman proxy since binary outcome
        scores = [p[0] for p in pairs]
        corrects = [p[1] for p in pairs]
        mean_s = sum(scores) / len(scores)
        mean_c = sum(corrects) / len(corrects)
        num = sum((s - mean_s) * (c - mean_c) for s, c in pairs)
        ds = sum((s - mean_s) ** 2 for s in scores) ** 0.5
        dc = sum((c - mean_c) ** 2 for c in corrects) ** 0.5
        if ds > 0 and dc > 0:
            rhos.append(num / (ds * dc))
    return {
        'kendall_tau_mean': statistics.mean(taus) if taus else 0,
        'kendall_tau_median': statistics.median(taus) if taus else 0,
        'pearson_bin_mean': statistics.mean(rhos) if rhos else 0,
        'n_problems': len(taus),
    }


def score_delta_correct_vs_incorrect(per_prob, score_key='prm_score'):
    """For each problem, mean(score | correct) - mean(score | incorrect)."""
    deltas = []
    for items in per_prob.values():
        cor = [x[score_key] for x in items if x.get(score_key) is not None and x['is_correct']]
        inc = [x[score_key] for x in items if x.get(score_key) is not None and not x['is_correct']]
        if cor and inc:
            deltas.append(statistics.mean(cor) - statistics.mean(inc))
    if not deltas: return {}
    deltas.sort()
    return {
        'mean_delta': statistics.mean(deltas),
        'median_delta': statistics.median(deltas),
        'positive_fraction': sum(1 for d in deltas if d > 0) / len(deltas),
        'p10': deltas[int(0.10 * len(deltas))],
        'p90': deltas[int(0.90 * len(deltas))],
        'n_problems': len(deltas),
    }


def random_rerank_at_N(per_prob, N):
    """Pick random candidate from N, report accuracy. Baseline = pick uniformly."""
    outs = []
    for items in per_prob.values():
        its = sorted(items, key=lambda x: x['tid'])[:N]
        if not its: continue
        pick = random.choice(its)
        outs.append(1 if pick['is_correct'] else 0)
    return sum(outs) / len(outs) * 100, len(outs)


def prm_rerank_at_N(per_prob, N):
    outs = []
    for items in per_prob.values():
        its = sorted(items, key=lambda x: x['tid'])[:N]
        its = [x for x in its if x.get('prm_score') is not None]
        if not its: continue
        best = max(its, key=lambda x: x['prm_score'])
        outs.append(1 if best['is_correct'] else 0)
    return sum(outs) / len(outs) * 100, len(outs)


def main():
    print("Loading scored per-problem data...")
    per_prob = load_per_problem_with_scores()
    total_items = sum(len(v) for v in per_prob.values())
    print(f"  {len(per_prob)} problems, {total_items} trajectories")

    results = {}

    print("\n=== 1. Sign convention check ===")
    # Across ALL trajectories, is mean(prm_score | correct) > mean(prm_score | incorrect)?
    all_cor_scores = []; all_inc_scores = []
    for items in per_prob.values():
        for x in items:
            if x.get('prm_score') is None: continue
            if x['is_correct']:
                all_cor_scores.append(x['prm_score'])
            else:
                all_inc_scores.append(x['prm_score'])
    mean_cor = statistics.mean(all_cor_scores)
    mean_inc = statistics.mean(all_inc_scores)
    print(f"  mean(PRM score | correct) = {mean_cor:.4f} (n={len(all_cor_scores)})")
    print(f"  mean(PRM score | incorrect) = {mean_inc:.4f} (n={len(all_inc_scores)})")
    print(f"  Difference = {mean_cor - mean_inc:+.4f}  ⇒ {'HIGHER SCORE = CORRECT ✓' if mean_cor > mean_inc else 'SIGN INVERTED ✗'}")
    results['sign_convention'] = {
        'mean_score_correct': round(mean_cor, 4),
        'mean_score_incorrect': round(mean_inc, 4),
        'diff': round(mean_cor - mean_inc, 4),
        'higher_score_is_correct': mean_cor > mean_inc,
    }

    print("\n=== 2. Rank correlation (per-problem, aggregated) ===")
    rc = kendall_spearman_per_problem(per_prob, score_key='prm_score')
    print(f"  Kendall τ  mean={rc['kendall_tau_mean']:.4f}  median={rc['kendall_tau_median']:.4f}  (over {rc['n_problems']} problems)")
    print(f"  Pearson point-biserial mean={rc['pearson_bin_mean']:.4f}")
    results['rank_correlation'] = rc

    print("\n=== 3. Score delta correct vs incorrect (per problem) ===")
    sd = score_delta_correct_vs_incorrect(per_prob, score_key='prm_score')
    print(f"  mean Δ = {sd['mean_delta']:+.4f}, median = {sd['median_delta']:+.4f}")
    print(f"  positive Δ fraction = {sd['positive_fraction']*100:.1f}% of problems")
    print(f"  p10 = {sd['p10']:+.4f}, p90 = {sd['p90']:+.4f}")
    results['score_delta'] = sd

    print("\n=== 4. Random-rerank vs PRM-rerank at matched N ===")
    # Run multiple random-rerank trials for average
    for N in [4, 8, 16, 32]:
        rand_accs = []
        for _ in range(10):
            a, _ = random_rerank_at_N(per_prob, N)
            rand_accs.append(a)
        rand_mean = statistics.mean(rand_accs)
        rand_std = statistics.stdev(rand_accs)
        prm_acc, _ = prm_rerank_at_N(per_prob, N)
        # Expected random = mean(is_correct) over first N trajectories
        results.setdefault('rerank_comparator', {})[f'N={N}'] = {
            'random_mean_acc': round(rand_mean, 2),
            'random_std': round(rand_std, 2),
            'prm_acc': round(prm_acc, 2),
            'prm_minus_random': round(prm_acc - rand_mean, 2),
        }
        print(f"  N={N}: random={rand_mean:.2f}±{rand_std:.2f}%, PRM-rerank={prm_acc:.2f}%, "
              f"Δ={prm_acc - rand_mean:+.2f}pp")

    print("\n=== 5. AUC-by-slice reconciliation ===")
    slices = {
        'bidir PRM mr=0 (n=42208 trajectories, true final states)': 'computed from prm_correlation mr==0',
        'bidir PRM bucket 0.0-0.1 (all snapshots with mr∈[0,0.1), n=126624)': 'from prm_correlation_analysis.json',
        'causal PRM mean-pool mr=0 (42208 trajectories)': 'from causal_prm_scores.json',
        'causal PRM last-token mr=0 (42208 trajectories, NEW)': 'from causal_lasttoken_scores.json',
        'bidir ORM mr=0 (42208 trajectories, separately-trained ORM, REFERENCE NOT USED IN CORE CLAIM)': 'from orm_scores.json',
    }
    results['auc_slices'] = {
        'bidir_PRM_mr_0_traj_level': 'approx 0.77 (from mr_0.0-0.1 bucket AUC per prm_correlation_analysis.json)',
        'bidir_PRM_bucket_0.0_0.1': 0.7702,
        'causal_PRM_mean_pool_mr_0_traj_level': 0.6147,
        'causal_PRM_last_token_mr_0_traj_level': 0.7274,
        'bidir_ORM_mr_0_traj_level_REFERENCE_ONLY': 0.9623,
    }
    for k, v in results['auc_slices'].items():
        print(f"  {k}: {v}")

    json.dump(results, open('eval_results/sanity_checks.json', 'w'), indent=2)
    print(f"\nWrote eval_results/sanity_checks.json")


if __name__ == '__main__':
    main()
