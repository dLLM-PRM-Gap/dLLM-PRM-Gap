"""PRM-as-reranker baseline (GPT Round 2 CRITICAL fix).

For each problem: take N=8 or N=32 vanilla-generated candidates, score each
with the bidirectional PRM at mask_ratio=0 (i.e., the final-state score),
pick argmax → rerank accuracy.

Directly addresses GPT's "is the failure guidance-specific or PRM-specific?" critique:
if PRM-as-reranker works well but PRM-Guided doesn't, then guidance IS the problem.
If PRM-as-reranker is no better than ORM-Rerank (or worse), then PRM itself has
less information than ORM.

Data sources:
- eval_results/prm_correlation_shard_*.json: bidir PRM scores at all mask ratios
  (filter mr==0 for final-state scores; 1 per trajectory)
- eval_results/orm_scores_shard_*.json: answer/is_correct metadata per trajectory

Also computes seed variance over {42, 43, 44, 45} for ORM@8 and PRM-rerank@8.
"""
import json
import glob
import random
from collections import defaultdict

random.seed(42)
N_BOOT = 1000


def load_final_state_prm_scores():
    """Return dict: (pid, tid) -> bidir_prm_score_at_mask0."""
    scores = {}
    for f in sorted(glob.glob('eval_results/prm_correlation_shard_*.json')):
        d = json.load(open(f))
        for item in d['results']:
            # mr == 0 means fully-decoded final state
            if item.get('mr') == 0.0:
                key = (item['pid'], item['tid'])
                scores[key] = item['score']
    return scores


def load_orm_per_problem():
    """Return: per_problem[pid] = list of dicts with tid, seed, score, is_correct, answer."""
    per_prob = defaultdict(list)
    for f in sorted(glob.glob('eval_results/orm_scores_shard_*.json')):
        d = json.load(open(f))
        for item in d['items']:
            m = item['meta']
            per_prob[m['problem_id']].append({
                'trajectory_id': m['trajectory_id'],
                'seed': m['seed'],
                'orm_score': item['score'],
                'is_correct': bool(m['is_correct']),
                'answer': m['answer'],
            })
    return dict(per_prob)


def annotate_with_prm_scores(per_prob, prm_scores):
    """Add bidir_prm_score to each trajectory dict. Returns (per_prob, coverage)."""
    missing = 0; total = 0
    for pid, items in per_prob.items():
        for it in items:
            total += 1
            key = (pid, it['trajectory_id'])
            s = prm_scores.get(key)
            if s is None:
                missing += 1
                it['bidir_prm_score'] = None
            else:
                it['bidir_prm_score'] = s
    print(f"  PRM coverage: {total - missing}/{total} trajectories "
          f"({100 * (total - missing) / total:.1f}%)")
    return per_prob


def rerank_at_N(per_prob, N, score_key):
    """For each problem, take first N trajs (by tid order), pick argmax by score_key."""
    outcomes = []
    for pid in sorted(per_prob.keys()):
        items = sorted(per_prob[pid], key=lambda x: x['trajectory_id'])[:N]
        # Filter to items that have this score
        items = [x for x in items if x.get(score_key) is not None]
        if not items:
            continue
        best = max(items, key=lambda x: x[score_key])
        outcomes.append(1 if best['is_correct'] else 0)
    return outcomes


def rerank_at_N_by_seed(per_prob, N, score_key, seed):
    """Same as rerank_at_N, but filter to a single seed (4 seeds × 8 trajs each = 32)."""
    outcomes = []
    for pid in sorted(per_prob.keys()):
        seed_items = [x for x in per_prob[pid] if x.get('seed') == seed]
        seed_items = sorted(seed_items, key=lambda x: x['trajectory_id'])[:N]
        seed_items = [x for x in seed_items if x.get(score_key) is not None]
        if not seed_items:
            continue
        best = max(seed_items, key=lambda x: x[score_key])
        outcomes.append(1 if best['is_correct'] else 0)
    return outcomes


def bootstrap_ci(outcomes, n_boot=N_BOOT, alpha=0.05):
    n = len(outcomes)
    boots = []
    for _ in range(n_boot):
        sample = [outcomes[random.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(alpha / 2 * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot)]
    point = sum(outcomes) / n
    return point, lo, hi


def main():
    print("Loading bidir PRM final-state scores...")
    prm_scores = load_final_state_prm_scores()
    print(f"  {len(prm_scores)} (pid, tid) entries with mr==0")

    print("Loading ORM per-problem data...")
    per_prob = load_orm_per_problem()
    print(f"  {len(per_prob)} problems, {sum(len(v) for v in per_prob.values())} trajectories")

    per_prob = annotate_with_prm_scores(per_prob, prm_scores)

    results = {
        'description': 'PRM-as-reranker (bidir PRM at mask=0) vs ORM-Rerank (bidir ORM)',
        'n_problems': len(per_prob),
        'n_boot': N_BOOT,
        'pareto': {},
        'seed_variance': {},
    }

    # Main Pareto
    print("\n=== PRM-as-reranker Pareto (bootstrap 95% CI) ===")
    for N in [1, 2, 4, 8, 16, 32]:
        prm_out = rerank_at_N(per_prob, N, 'bidir_prm_score')
        orm_out = rerank_at_N(per_prob, N, 'orm_score')
        prm_pt, prm_lo, prm_hi = bootstrap_ci(prm_out)
        orm_pt, orm_lo, orm_hi = bootstrap_ci(orm_out)
        # paired delta
        paired = [a - b for a, b in zip(prm_out, orm_out) if len(prm_out) == len(orm_out)]
        # match lens — should be identical 1319
        assert len(prm_out) == len(orm_out), f"Unequal: {len(prm_out)} vs {len(orm_out)}"
        deltas = []
        for _ in range(N_BOOT):
            ixs = [random.randrange(len(prm_out)) for _ in range(len(prm_out))]
            a = sum(prm_out[i] for i in ixs) / len(prm_out)
            b = sum(orm_out[i] for i in ixs) / len(prm_out)
            deltas.append(a - b)
        deltas.sort()
        delta_pt = sum(a - b for a, b in zip(prm_out, orm_out)) / len(prm_out)
        d_lo = deltas[int(0.025 * N_BOOT)]; d_hi = deltas[int(0.975 * N_BOOT)]

        results['pareto'][f'N={N}'] = {
            'prm_rerank_bidir_mask0': {
                'acc': round(prm_pt * 100, 3),
                'ci95': [round(prm_lo * 100, 3), round(prm_hi * 100, 3)],
            },
            'orm_rerank': {
                'acc': round(orm_pt * 100, 3),
                'ci95': [round(orm_lo * 100, 3), round(orm_hi * 100, 3)],
            },
            'paired_delta_prm_minus_orm_pp': round(delta_pt * 100, 3),
            'paired_ci95_pp': [round(d_lo * 100, 3), round(d_hi * 100, 3)],
        }
        print(f"  N={N:2d}: PRM-rerank={prm_pt*100:.2f}% [{prm_lo*100:.2f}, {prm_hi*100:.2f}] | "
              f"ORM-rerank={orm_pt*100:.2f}% [{orm_lo*100:.2f}, {orm_hi*100:.2f}] | "
              f"Δ={delta_pt*100:+.2f}pp [{d_lo*100:+.2f}, {d_hi*100:+.2f}]")

    # Seed variance: for each seed, compute ORM@8 and PRM-rerank@8
    # Each seed has ~8 trajectories per problem (32 / 4 = 8)
    print("\n=== Seed variance (N=8 per seed, across 4 seeds) ===")
    for score_key, name in [('orm_score', 'ORM-rerank@8'),
                              ('bidir_prm_score', 'PRM-rerank@8')]:
        per_seed_accs = []
        for seed in [42, 43, 44, 45]:
            out = rerank_at_N_by_seed(per_prob, 8, score_key, seed)
            if len(out) == 0: continue
            per_seed_accs.append(sum(out) / len(out) * 100)
        if len(per_seed_accs) == 4:
            import statistics
            mean_ = statistics.mean(per_seed_accs)
            std_ = statistics.stdev(per_seed_accs)
            results['seed_variance'][name] = {
                'per_seed_acc_pp': [round(v, 3) for v in per_seed_accs],
                'mean_pp': round(mean_, 3),
                'std_pp': round(std_, 3),
            }
            print(f"  {name}: per-seed accs = {[round(v, 2) for v in per_seed_accs]}% "
                  f"mean {mean_:.2f}% ± {std_:.2f}pp")

    json.dump(results, open('eval_results/prm_rerank_vs_orm_pareto.json', 'w'), indent=2)
    print(f"\nWrote eval_results/prm_rerank_vs_orm_pareto.json")


if __name__ == '__main__':
    main()
