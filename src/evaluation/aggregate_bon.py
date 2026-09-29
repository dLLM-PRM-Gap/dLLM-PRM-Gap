"""
Best-of-N aggregation for dLLM temperature sweep diagnostic.

Given N result JSONs from eval_baselines.py (same task, same temperature, different seeds),
compute:
  1. Individual accuracy per seed
  2. Majority@N: majority voting across N samples per problem
  3. Oracle@N: any-correct across N samples per problem
  4. Diversity: fraction of problems where not all N answers agree

Usage:
    python aggregate_bon.py --results_dir eval_results/bon_diagnostic/ \
        --task gsm8k --temperature 0.8 --num_steps 128

    python aggregate_bon.py --files result_s0.json result_s1.json result_s2.json result_s3.json
"""

import os
import sys
import json
import argparse
import re
from collections import Counter
from pathlib import Path


def normalize_answer(answer_str, task):
    """Normalize an extracted answer for comparison."""
    if answer_str is None:
        return None
    s = str(answer_str).strip()
    if not s:
        return None

    if task in ("gsm8k", "math500"):
        # Remove commas, dollar signs, percent signs
        s = s.replace(",", "").replace("$", "").replace("%", "")
        # Try to parse as number
        try:
            val = float(s)
            # Convert to int if it's a whole number
            if val == int(val):
                return str(int(val))
            return f"{val:.6f}"
        except ValueError:
            return s.lower().strip()
    else:
        # For other tasks, just lowercase and strip
        return s.lower().strip()


def extract_answer_from_sample(sample, task):
    """Extract the predicted answer from an lm-eval sample dict."""
    # lm-eval stores filtered responses after metric-specific extraction
    filtered = sample.get("filtered_resps")
    if filtered and isinstance(filtered, list) and len(filtered) > 0:
        ans = filtered[0]
        if isinstance(ans, list) and len(ans) > 0:
            ans = ans[0]
        # lm-eval marks extraction failures as '[invalid]'
        if isinstance(ans, str) and ans.strip() not in ('[invalid]', ''):
            return normalize_answer(ans, task)

    # Fallback: try to extract from raw response
    resps = sample.get("resps")
    if resps and isinstance(resps, list) and len(resps) > 0:
        resp = resps[0]
        if isinstance(resp, list) and len(resp) > 0:
            resp = resp[0]
        if isinstance(resp, str):
            if task in ("gsm8k", "math500"):
                match = re.search(r'####\s*([\-\d,\.]+)', resp)
                if match:
                    return normalize_answer(match.group(1), task)
                numbers = re.findall(r'[\-\d,\.]+', resp)
                if numbers:
                    return normalize_answer(numbers[-1], task)
            elif task == "math500":
                match = re.search(r'\\boxed\{([^}]+)\}', resp)
                if match:
                    return normalize_answer(match.group(1), task)
    return None


def extract_target(sample, task):
    """Extract the ground truth answer from an lm-eval sample dict."""
    # For GSM8K/MATH, target is the full solution text — extract #### N
    target = sample.get("target", "")
    if task == "gsm8k":
        match = re.search(r'####\s*([\-\d,\.]+)', target)
        if match:
            return normalize_answer(match.group(1), task)
    elif task == "math500":
        match = re.search(r'\\boxed\{([^}]+)\}', target)
        if match:
            return normalize_answer(match.group(1), task)
        # Fallback: try the whole target
        return normalize_answer(target, task)

    # For other tasks or fallback
    if target:
        return normalize_answer(target, task)

    # Try doc field
    doc = sample.get("doc", {})
    if task == "gsm8k":
        answer = doc.get("answer", "")
        match = re.search(r'####\s*([\-\d,\.]+)', answer)
        if match:
            return normalize_answer(match.group(1), task)
    return None


def is_sample_correct(sample):
    """Check if a sample was answered correctly using lm-eval's exact_match field."""
    em = sample.get("exact_match")
    if em is not None:
        return float(em) > 0.5
    return False


def load_results(path):
    """Load a result JSON and return its samples."""
    with open(path) as f:
        data = json.load(f)
    return data


def parse_filename(filename, task):
    """Parse eval_baselines output filename to extract config.

    Filename format: vanilla_{task}_steps{N}[_temp{T}][_at{AT}][_tp{TP}]_seed{S}[_tag].json
    Defaults: temp=0.2, alg_temp=0.0, top_p=0.95
    """
    name = Path(filename).stem
    config = {"temp": 0.2, "alg_temp": 0.0, "top_p": 0.95, "seed": None, "tag": None}

    m = re.search(r'_temp([\d.]+)', name)
    if m:
        config["temp"] = float(m.group(1))
    m = re.search(r'_at([\d.]+)', name)
    if m:
        config["alg_temp"] = float(m.group(1))
    m = re.search(r'_tp([\d.]+)', name)
    if m:
        config["top_p"] = float(m.group(1))
    m = re.search(r'_seed(\d+)', name)
    if m:
        config["seed"] = int(m.group(1))
    m = re.search(r'_bon_(c\d+)', name)
    if m:
        config["tag"] = m.group(1)
    return config


def find_result_files(results_dir, task, temperature=None, num_steps=128, tag_prefix=None):
    """Find all result JSONs matching the given criteria."""
    files = []
    base_pattern = f"vanilla_{task}_steps{num_steps}"
    for f in sorted(Path(results_dir).glob("*.json")):
        if base_pattern not in f.name:
            continue
        if tag_prefix and f"bon_{tag_prefix}" not in f.name:
            continue
        if temperature is not None:
            cfg = parse_filename(f.name, task)
            if abs(cfg["temp"] - temperature) > 0.001:
                continue
        files.append(str(f))
    return files


def auto_group_by_config(results_dir, task, num_steps=128):
    """Auto-discover all result files and group by (temp, alg_temp, top_p)."""
    base_pattern = f"vanilla_{task}_steps{num_steps}"
    groups = {}
    for f in sorted(Path(results_dir).glob("*.json")):
        if base_pattern not in f.name:
            continue
        cfg = parse_filename(f.name, task)
        key = (cfg["temp"], cfg["alg_temp"], cfg["top_p"])
        tag = cfg.get("tag", "")
        group_key = tag if tag else f"t{key[0]}_at{key[1]}_tp{key[2]}"
        if key not in groups:
            groups[key] = {"files": [], "tag": tag}
        groups[key]["files"].append(str(f))
    return groups


def aggregate(result_files, task):
    """Aggregate N result files and compute Best-of-N metrics."""
    all_data = []
    individual_accs = []

    for path in result_files:
        data = load_results(path)
        all_data.append(data)

        # Compute individual accuracy
        if "accuracies" in data and task in data["accuracies"]:
            metrics = data["accuracies"][task]
            # Find the primary metric
            for key in ["exact_match,flexible-extract", "math_verify,none", "pass@1,create_test"]:
                if key in metrics:
                    individual_accs.append(metrics[key])
                    break

    N = len(result_files)
    print(f"\n{'='*60}")
    print(f"Best-of-N Aggregation: task={task}, N={N}")
    print(f"{'='*60}")

    # Print individual accuracies
    for i, (path, acc) in enumerate(zip(result_files, individual_accs)):
        seed = Path(path).stem.split("seed")[-1].split("_")[0]
        print(f"  Seed {seed}: {acc*100:.2f}%")

    if individual_accs:
        print(f"  Mean individual: {sum(individual_accs)/len(individual_accs)*100:.2f}%")

    # Now do per-problem aggregation using samples
    # Check if samples are available
    samples_available = all(("samples" in d) for d in all_data)
    if not samples_available:
        print("\n  WARNING: 'samples' not found in result JSONs.")
        print("  Cannot compute majority/oracle metrics.")
        print("  Only individual accuracies are available.")
        return {
            "N": N,
            "individual_accs": individual_accs,
            "mean_individual": sum(individual_accs) / len(individual_accs) if individual_accs else None,
        }

    # Get the task key in samples
    # lm-eval uses the full task name as key
    task_key = None
    for key in all_data[0]["samples"]:
        task_key = key
        break

    if task_key is None:
        print("  WARNING: No task key found in samples.")
        return {"N": N, "individual_accs": individual_accs}

    # Handle dual-filter lm-eval output: GSM8K stores both strict-match and
    # flexible-extract entries (2x samples). We only use flexible-extract.
    raw_samples_0 = all_data[0]["samples"][task_key]
    filter_types = set(s.get("filter") for s in raw_samples_0)
    preferred_filter = "flexible-extract" if "flexible-extract" in filter_types else None

    if preferred_filter and len(filter_types) > 1:
        # Filter all datasets to only use the preferred filter
        for data in all_data:
            data["samples"][task_key] = [
                s for s in data["samples"][task_key]
                if s.get("filter") == preferred_filter
            ]
        print(f"  Filter: using '{preferred_filter}' (dropped {len(filter_types)-1} other filter(s))")

    # Align samples by doc_id
    num_problems = len(all_data[0]["samples"][task_key])
    print(f"  Problems: {num_problems}")

    majority_correct = 0
    oracle_correct = 0
    all_agree = 0
    total = 0

    for prob_idx in range(num_problems):
        answers = []
        corrects = []
        target = None

        for data in all_data:
            sample = data["samples"][task_key][prob_idx]
            pred = extract_answer_from_sample(sample, task)
            answers.append(pred)

            # Extract target once
            if target is None:
                target = extract_target(sample, task)

            # Use our own comparison (consistent with majority voting logic)
            # Also check lm-eval's exact_match as backup
            own_correct = (pred is not None and target is not None and pred == target)
            lm_correct = is_sample_correct(sample)
            corrects.append(own_correct or lm_correct)

        total += 1

        # Oracle: any correct?
        if any(corrects):
            oracle_correct += 1

        # Majority voting
        valid_answers = [a for a in answers if a is not None]
        if valid_answers:
            counter = Counter(valid_answers)
            majority_answer = counter.most_common(1)[0][0]
            if target is not None and majority_answer == target:
                majority_correct += 1

        # Diversity: do all answers agree?
        unique_answers = set(a for a in answers if a is not None)
        if len(unique_answers) <= 1:
            all_agree += 1

    majority_acc = majority_correct / total if total > 0 else 0
    oracle_acc = oracle_correct / total if total > 0 else 0
    diversity = 1.0 - (all_agree / total) if total > 0 else 0

    print(f"\n  Results:")
    print(f"  Majority@{N}:  {majority_acc*100:.2f}% ({majority_correct}/{total})")
    print(f"  Oracle@{N}:    {oracle_acc*100:.2f}% ({oracle_correct}/{total})")
    print(f"  Diversity:     {diversity*100:.1f}% problems with disagreement")
    print(f"{'='*60}\n")

    return {
        "N": N,
        "task": task,
        "individual_accs": individual_accs,
        "mean_individual": sum(individual_accs) / len(individual_accs) if individual_accs else None,
        "majority_acc": majority_acc,
        "oracle_acc": oracle_acc,
        "diversity": diversity,
        "num_problems": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Best-of-N aggregation for dLLM diagnostic")
    parser.add_argument("--files", nargs="+", help="Explicit list of result JSON files")
    parser.add_argument("--results_dir", type=str, help="Directory to scan for result files")
    parser.add_argument("--task", type=str, required=True, choices=["gsm8k", "math500", "humaneval"])
    parser.add_argument("--temperature", type=float, default=None, help="Filter by temperature")
    parser.add_argument("--tag_prefix", type=str, default=None, help="Filter by tag prefix (e.g., 'c2' for bon_c2)")
    parser.add_argument("--num_steps", type=int, default=128, help="Filter by num_steps")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    if args.files:
        result_files = args.files
    elif args.results_dir:
        if args.temperature is None and args.tag_prefix is None:
            # Auto-discover and group by (temp, alg_temp, top_p)
            groups = auto_group_by_config(args.results_dir, args.task, args.num_steps)
            all_results = {}
            for key, info in sorted(groups.items()):
                temp, alg_temp, top_p = key
                tag = info["tag"] or f"t{temp}_at{alg_temp}_tp{top_p}"
                label = f"temp={temp}, alg_temp={alg_temp}, top_p={top_p}"
                if info["tag"]:
                    label += f" [{info['tag']}]"
                print(f"\n--- {label} ({len(info['files'])} files) ---")
                for f in info['files']:
                    print(f"  {Path(f).name}")
                result = aggregate(info['files'], args.task)
                result["config"] = {"temp": temp, "alg_temp": alg_temp, "top_p": top_p, "tag": info["tag"]}
                all_results[tag] = result

            # Print summary table
            if all_results:
                print("\n" + "=" * 95)
                print(f"SUMMARY: {args.task} Best-of-N Config Sweep (steps={args.num_steps})")
                print("=" * 95)
                print(f"{'Config':>6} | {'temp':>5} | {'alg_t':>5} | {'top_p':>5} | {'N':>3} | {'Mean Indiv':>11} | {'Majority@N':>11} | {'Oracle@N':>10} | {'Diversity':>10}")
                print("-" * 95)
                for tag, r in sorted(all_results.items()):
                    cfg = r.get("config", {})
                    t = f"{cfg.get('temp', '?')}"
                    at = f"{cfg.get('alg_temp', '?')}"
                    tp = f"{cfg.get('top_p', '?')}"
                    ctag = cfg.get('tag') or tag
                    mi = f"{r.get('mean_individual', 0)*100:.2f}%" if r.get('mean_individual') else "N/A"
                    maj = f"{r.get('majority_acc', 0)*100:.2f}%" if 'majority_acc' in r else "N/A"
                    orc = f"{r.get('oracle_acc', 0)*100:.2f}%" if 'oracle_acc' in r else "N/A"
                    div = f"{r.get('diversity', 0)*100:.1f}%" if 'diversity' in r else "N/A"
                    print(f"{ctag:>6} | {t:>5} | {at:>5} | {tp:>5} | {r['N']:>3} | {mi:>11} | {maj:>11} | {orc:>10} | {div:>10}")
                print("=" * 95)

                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(all_results, f, indent=2)
                    print(f"Saved to {args.output}")
            return

        result_files = find_result_files(
            args.results_dir, args.task,
            temperature=args.temperature,
            num_steps=args.num_steps,
            tag_prefix=args.tag_prefix,
        )
    else:
        print("ERROR: Provide either --files or --results_dir")
        sys.exit(1)

    if not result_files:
        print("ERROR: No result files found")
        sys.exit(1)

    print(f"Found {len(result_files)} result files:")
    for f in result_files:
        print(f"  {f}")

    result = aggregate(result_files, args.task)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
