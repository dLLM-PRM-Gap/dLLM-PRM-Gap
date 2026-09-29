"""Post-hoc analysis: two-pass causal vs bidir PRM.

Aggregates eval_results/two_pass_causal_shard_*.json (8 shards),
computes AUC of causal_fwd, causal_bwd, causal_twopass_avg, bidir
against final_correct, both overall and per mask_ratio bucket.

Writes eval_results/two_pass_causal_analysis.json.
"""
import json
import glob
import numpy as np
from sklearn.metrics import roc_auc_score

SHARD_GLOB = "eval_results/two_pass_causal_shard_*.json"
OUT = "eval_results/two_pass_causal_analysis.json"

def bucket(mr):
    for lo, hi, name in [
        (0.0, 0.10, "0.0-0.10"),
        (0.1, 0.30, "0.1-0.30"),
        (0.3, 0.50, "0.3-0.50"),
        (0.5, 0.70, "0.5-0.70"),
        (0.7, 0.90, "0.7-0.90"),
        (0.9, 1.01, "0.9-1.01"),
    ]:
        if lo <= mr < hi:
            return name
    return "other"

def main():
    all_rows = []
    for f in sorted(glob.glob(SHARD_GLOB)):
        obj = json.load(open(f))
        all_rows.extend(obj["results"])
    print(f"Loaded {len(all_rows)} rows across shards.")
    keys = ["causal_fwd", "causal_bwd", "causal_twopass_avg", "bidir"]

    y = np.array([bool(r["final_correct"]) for r in all_rows], dtype=np.int8)
    mr = np.array([r["mr"] for r in all_rows], dtype=np.float32)
    scores = {k: np.array([r[k] for r in all_rows], dtype=np.float32) for k in keys}

    def safe_auc(y, s, mask=None):
        if mask is not None:
            y, s = y[mask], s[mask]
        if len(y) < 10 or y.sum() == 0 or y.sum() == len(y):
            return None
        return float(roc_auc_score(y, s))

    overall = {k: safe_auc(y, scores[k]) for k in keys}

    buckets = {}
    for b in ["0.0-0.10", "0.1-0.30", "0.3-0.50", "0.5-0.70", "0.7-0.90", "0.9-1.01"]:
        lo, hi = [float(x) for x in b.split("-")]
        mask = (mr >= lo) & (mr < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        buckets[b] = {
            "n": n,
            "pos_rate": float(y[mask].mean()),
            "auc": {k: safe_auc(y, scores[k], mask) for k in keys},
        }

    correlations = {
        "fwd_vs_bwd_pearson": float(np.corrcoef(scores["causal_fwd"], scores["causal_bwd"])[0, 1]),
        "fwd_vs_bidir_pearson": float(np.corrcoef(scores["causal_fwd"], scores["bidir"])[0, 1]),
        "twopass_vs_bidir_pearson": float(np.corrcoef(scores["causal_twopass_avg"], scores["bidir"])[0, 1]),
    }

    out = {
        "n_rows": len(all_rows),
        "overall_auc": overall,
        "per_mask_bucket": buckets,
        "score_correlations": correlations,
        "conclusion_auto": _conclusion(overall, buckets),
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"Wrote {OUT}")
    print("\nOverall AUC:")
    for k, v in overall.items():
        print(f"  {k:22s} {v:.4f}" if v is not None else f"  {k:22s} n/a")
    print("\nPer-bucket AUC (causal_twopass_avg → bidir):")
    for b, d in buckets.items():
        cf = d["auc"]["causal_fwd"]
        cb = d["auc"]["causal_bwd"]
        ta = d["auc"]["causal_twopass_avg"]
        bi = d["auc"]["bidir"]
        gap = (bi - ta) if (ta is not None and bi is not None) else None
        print(f"  {b:9s} n={d['n']:7d} fwd={cf:.3f} bwd={cb:.3f} 2p_avg={ta:.3f} bidir={bi:.3f} gap(bidir-2p)={gap:+.3f}" if gap is not None else f"  {b:9s} n={d['n']:7d} gap=n/a")
    print(f"\nCorrelations: {correlations}")

def _conclusion(overall, buckets):
    ta = overall["causal_twopass_avg"]
    bi = overall["bidir"]
    if ta is None or bi is None:
        return "n/a"
    gap = bi - ta
    if gap < 0.02:
        return f"Two-pass causal RECOVERS bidir (gap={gap:+.3f}) → mechanism is side-of-context"
    else:
        return f"Two-pass causal FAILS to recover bidir (gap={gap:+.3f}) → mechanism is NOT side-of-context; it's joint-unordered modeling"

if __name__ == "__main__":
    main()
