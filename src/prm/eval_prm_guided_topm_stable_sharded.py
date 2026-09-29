"""Audited sharded evaluator for stable deterministic top-1/top-M guidance.

Formal runs are fixed to GSM8K test, K=8, T=128, a checkpoint after step 64,
and M in {1, 2}.  Every candidate stream is derived from
``(base_seed, global_problem_id, particle_id, stage)`` and is therefore
independent of shard count, traversal order, and process-level RNG state.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch


RELEASE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("DIFFUSION_PRM_ROOT", RELEASE_ROOT)).resolve()
for candidate in (
    PROJECT_ROOT / "external" / "CoDD" / "eval",
    PROJECT_ROOT / "external" / "CoDD",
    RELEASE_ROOT.parent / "external" / "CoDD" / "eval",
    RELEASE_ROOT.parent / "external" / "CoDD",
):
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from .checkpointing import load_diffusion_prm  # noqa: E402
from .eval_prm_guided_sharded import (  # noqa: E402
    build_gsm8k_prompt,
    extract_gsm8k_answer,
    normalize_answer,
)
from .eval_prm_smc_sharded import (  # noqa: E402
    load_gsm8k,
    parse_problem_ids,
    validate_submitted_main_gate,
)
from .matched_continuation import (  # noqa: E402
    ROLES as MATCHED_SCORER_ROLES,
    validate_matched_checkpoint_gate,
)
from .prm_guided_topm_stable import (  # noqa: E402
    StableTopMConfig,
    prm_guided_topm_stable_sample,
)
from .protocol import (  # noqa: E402
    artifact_descriptor,
    atomic_write_json,
    build_protocol_manifest,
    sha256_text,
    validate_artifact_attestation,
)


LOGGER = logging.getLogger("eval_prm_guided_topm_stable")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)


def _load_json_mapping(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    with open(resolved, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _mean(results: list[Mapping[str, Any]], field) -> float:
    if not results:
        raise ValueError("cannot summarize zero results")
    return sum(float(field(item)) for item in results) / len(results)


def summarize_results(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("cannot summarize zero results")
    call_totals = {
        key: sum(int(item["call_counts"][key]) for item in results)
        for key in results[0]["call_counts"]
    }
    return {
        "num_problems": len(results),
        "accuracy": _mean(results, lambda item: item["readout"]["correct"]),
        "oracle_at_8": _mean(results, lambda item: item["oracle_correct"]),
        "mean_unique_valid_extracted_answers": _mean(
            results, lambda item: item["unique_valid_extracted_answers"]
        ),
        "mean_invalid_candidate_count": _mean(
            results, lambda item: item["invalid_candidate_count"]
        ),
        "mean_wall_time_seconds": _mean(
            results, lambda item: item["wall_time_seconds"]
        ),
        "call_count_totals": call_totals,
    }


def evaluate_one(
    *,
    example: Mapping[str, Any],
    tokenizer,
    generation_model,
    prm,
    base_seed: int,
    baseline_config: StableTopMConfig,
) -> dict[str, Any]:
    question = str(example["question"])
    prompt_text = build_gsm8k_prompt(question)
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to("cuda")
    prompt_length = input_ids.shape[-1]
    started = time.time()
    terminal_states, baseline_info = prm_guided_topm_stable_sample(
        generation_model,
        prm,
        input_ids,
        base_seed=base_seed,
        global_problem_id=example["global_problem_id"],
        config=baseline_config,
        strict_fixed_protocol=True,
    )
    elapsed = time.time() - started

    gold = normalize_answer(example["gold_answer"])
    terminal = baseline_info["terminal"]
    parent_ids = [int(value) for value in terminal["parent_stage1_particle_ids"]]
    terminal_seeds = [int(value) for value in terminal["row_seeds"]]
    terminal_logits = [float(value) for value in terminal["raw_prm_logits"]]
    candidates = []
    for particle_id in range(baseline_config.particles):
        generated_ids = (
            terminal_states[particle_id, prompt_length:].detach().cpu().tolist()
        )
        generated_text = tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )
        extracted = normalize_answer(extract_gsm8k_answer(generated_text))
        candidates.append(
            {
                "particle_id": particle_id,
                "parent_stage1_particle_id": parent_ids[particle_id],
                "stage2_seed": terminal_seeds[particle_id],
                "generated_token_ids": generated_ids,
                "generated_text": generated_text,
                "extracted_answer": extracted,
                "correct": extracted == gold,
                "terminal_raw_prm_logit": terminal_logits[particle_id],
            }
        )

    selected_id = int(terminal["selected_particle_id"])
    selected = candidates[selected_id]
    valid_answers = {
        item["extracted_answer"]
        for item in candidates
        if item["extracted_answer"] is not None
    }
    expected_counts = {
        "logical_denoising": 1024,
        "logical_prm_scoring": 16,
        "logical_total": 1040,
        "physical_denoising_forwards": 128,
        "physical_prm_forwards": 2,
    }
    if baseline_info["call_counts"] != expected_counts:
        raise AssertionError(
            "matched-compute assertion failed: "
            f"{baseline_info['call_counts']} != {expected_counts}"
        )
    return {
        "global_problem_id": int(example["global_problem_id"]),
        "question": question,
        "gold_answer": gold,
        "prompt_sha256": sha256_text(prompt_text),
        "prompt_length": prompt_length,
        "base_seed": int(base_seed),
        "top_m": baseline_config.top_m,
        "readout": {
            "particle_id": selected_id,
            "selected_answer": selected["extracted_answer"],
            "raw_terminal_prm_logit": selected["terminal_raw_prm_logit"],
            "correct": selected["correct"],
            "tie_break": "descending_logit_then_lowest_particle_id",
        },
        "oracle_correct": any(item["correct"] for item in candidates),
        "unique_valid_extracted_answers": len(valid_answers),
        "invalid_candidate_count": sum(
            item["extracted_answer"] is None for item in candidates
        ),
        "candidates": candidates,
        "stage1": baseline_info["stage1"],
        "terminal": terminal,
        "call_counts": baseline_info["call_counts"],
        "wall_time_seconds": elapsed,
    }


def shard_filename(args: argparse.Namespace, shard_id: int) -> str:
    return (
        f"topm_gsm8k_{args.split}_K8_M{args.top_m}_s{args.seed}"
        f"_shard{shard_id:03d}-of-{args.num_shards:03d}.json"
    )


def _manifest_lock(manifest: Mapping[str, Any]) -> dict[str, Any]:
    extra = manifest.get("extra", {})
    git = manifest.get("git", {})
    return {
        "stable_seed_spec": manifest.get("stable_seed_spec"),
        "source_commit": git.get("commit"),
        "source_dirty": git.get("dirty"),
        "artifacts": manifest.get("artifacts"),
        "prompt_template_sha256": manifest.get("prompt_template_sha256"),
        "extraction_callable_sha256": manifest.get(
            "extraction_callable_sha256"
        ),
        "dataset_name": extra.get("dataset_name"),
        "dataset_subset": extra.get("dataset_subset"),
        "dataset_split": extra.get("dataset_split"),
        "dataset_fingerprint": extra.get("dataset_fingerprint"),
        "selected_question_list_sha256": extra.get(
            "selected_question_list_sha256"
        ),
        "submitted_main_checkpoint_gate": extra.get(
            "submitted_main_checkpoint_gate"
        ),
        "matched_prm_checkpoint_gate": extra.get(
            "matched_prm_checkpoint_gate"
        ),
        "prm_scorer_role": extra.get("prm_scorer_role", "submitted"),
        "fixed_protocol": extra.get("fixed_protocol"),
    }


def _validate_resume_payload(
    existing: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    shard_ids: list[int],
    current_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    expected_metadata = {
        "task": "gsm8k",
        "split": args.split,
        "method": "stable_prm_guided_topm",
        "seed": args.seed,
        "top_m": args.top_m,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "selected_problem_ids": shard_ids,
    }
    errors = [
        f"{key} mismatch"
        for key, value in expected_metadata.items()
        if existing.get(key) != value
    ]
    previous_manifest = existing.get("manifest")
    if not isinstance(previous_manifest, Mapping):
        errors.append("missing prior manifest")
    elif _manifest_lock(previous_manifest) != _manifest_lock(current_manifest):
        errors.append("prior manifest lock differs from current run")
    results = existing.get("per_problem")
    if not isinstance(results, list):
        errors.append("per_problem is not a list")
        results = []
    ids = [int(item["global_problem_id"]) for item in results]
    if len(ids) != len(set(ids)):
        errors.append("duplicate completed global_problem_id")
    if not set(ids).issubset(shard_ids):
        errors.append("completed ids are outside this shard")
    if errors:
        raise ValueError("unsafe resume payload: " + "; ".join(errors))
    return list(results), previous_manifest


def _validate_shard_manifest_set(manifests: list[Mapping[str, Any]]) -> None:
    if not manifests:
        raise ValueError("no shard manifests")
    expected = _manifest_lock(manifests[0])
    for index, manifest in enumerate(manifests[1:], start=1):
        if _manifest_lock(manifest) != expected:
            raise ValueError(f"shard {index} protocol manifest does not match shard 0")


def merge_shards(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    results: list[dict[str, Any]] = []
    manifests: list[Mapping[str, Any]] = []
    for shard_id in range(args.num_shards):
        path = output_dir / shard_filename(args, shard_id)
        if not path.is_file():
            raise FileNotFoundError(f"missing top-M shard: {path}")
        with open(path, encoding="utf-8") as handle:
            shard = json.load(handle)
        locks = {
            "task": "gsm8k",
            "split": args.split,
            "method": "stable_prm_guided_topm",
            "seed": args.seed,
            "top_m": args.top_m,
            "shard_id": shard_id,
            "num_shards": args.num_shards,
        }
        for key, expected in locks.items():
            if shard.get(key) != expected:
                raise ValueError(
                    f"{path}: {key}={shard.get(key)!r}, expected {expected!r}"
                )
        if args.formal_run:
            expected_ids = list(range(1319))[shard_id :: args.num_shards]
            if shard.get("selected_problem_ids") != expected_ids:
                raise ValueError(f"{path}: formal shard id allocation mismatch")
        results.extend(shard["per_problem"])
        manifests.append(shard["manifest"])

    _validate_shard_manifest_set(manifests)
    if args.formal_run:
        for manifest in manifests:
            git = manifest.get("git", {})
            if not git.get("commit") or git.get("dirty") is not False:
                raise ValueError("formal top-M merge requires clean shard manifests")
            extra = manifest.get("extra", {})
            artifacts = manifest.get("artifacts", {})
            scorer_role = extra.get("prm_scorer_role", "submitted")
            if scorer_role == "submitted":
                gate = extra.get("submitted_main_checkpoint_gate")
                required_artifacts = (
                    "submitted_prm",
                    "submitted_prm_provenance",
                    "full_to_adapter_equivalence",
                )
            else:
                if scorer_role not in MATCHED_SCORER_ROLES:
                    raise ValueError("formal top-M merge has unknown scorer role")
                gate = extra.get("matched_prm_checkpoint_gate")
                required_artifacts = (
                    "matched_prm",
                    "matched_prm_provenance",
                    "matched_prm_pair_report",
                )
            if (
                not isinstance(gate, Mapping)
                or gate.get("all_passed") is not True
                or any(
                    not artifacts.get(name, {}).get("sha256")
                    for name in required_artifacts
                )
            ):
                raise ValueError(
                    "formal top-M merge lacks its scorer gate/artifact locks"
                )
    ids = [int(item["global_problem_id"]) for item in results]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate global_problem_id across top-M shards")
    results.sort(key=lambda item: item["global_problem_id"])
    ids = [int(item["global_problem_id"]) for item in results]
    if args.expected_problems is not None and len(results) != args.expected_problems:
        raise ValueError(
            f"merged {len(results)} problems, expected {args.expected_problems}"
        )
    if args.formal_run and ids != list(range(1319)):
        raise ValueError("formal top-M merge does not cover exactly ids 0..1318")

    merged = {
        "schema_version": 1,
        "task": "gsm8k",
        "split": args.split,
        "method": "stable_prm_guided_topm",
        "seed": args.seed,
        "top_m": args.top_m,
        "num_shards": args.num_shards,
        "summary": summarize_results(results),
        "per_problem": results,
        "shard_manifests": manifests,
    }
    destination = output_dir / (
        f"topm_gsm8k_{args.split}_K8_M{args.top_m}_s{args.seed}_merged.json"
    )
    atomic_write_json(destination, merged)
    LOGGER.info(
        "MERGED M=%d seed=%d: %d problems, accuracy %.2f%%, oracle %.2f%%",
        args.top_m,
        args.seed,
        len(results),
        100 * merged["summary"]["accuracy"],
        100 * merged["summary"]["oracle_at_8"],
    )
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--top_m", type=int, choices=(1, 2), required=True)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--prm_checkpoint", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--problem_ids", default=None)
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--merge_only", action="store_true")
    parser.add_argument("--expected_problems", type=int, default=None)
    parser.add_argument("--formal_run", action="store_true")
    parser.add_argument("--model_sha256", default=None)
    parser.add_argument("--model_hash_attestation", default=None)
    parser.add_argument("--checkpoint_sha256", default=None)
    parser.add_argument("--source_checkpoint_sha256", default=None)
    parser.add_argument("--checkpoint_provenance", default=None)
    parser.add_argument("--checkpoint_provenance_sha256", default=None)
    parser.add_argument("--checkpoint_equivalence_report", default=None)
    parser.add_argument("--checkpoint_pair_report", default=None)
    parser.add_argument("--checkpoint_pair_report_sha256", default=None)
    parser.add_argument(
        "--prm_scorer_role",
        choices=("submitted", *sorted(MATCHED_SCORER_ROLES)),
        default="submitted",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("shard_id must lie in [0, num_shards)")
    if args.save_every <= 0:
        raise ValueError("save_every must be positive")
    if not args.merge_only and (not args.model_path or not args.prm_checkpoint):
        raise ValueError("evaluation requires --model_path and --prm_checkpoint")
    StableTopMConfig(top_m=args.top_m).validate(strict_fixed_protocol=True)
    if args.formal_run:
        if args.split != "test":
            raise ValueError("formal top-M runs must use GSM8K test")
        if args.problem_ids is not None or args.max_problems is not None:
            raise ValueError("formal top-M runs must cover the full test set")
        if not args.merge_only:
            common = (
                args.model_sha256,
                args.model_hash_attestation,
                args.checkpoint_sha256,
                args.checkpoint_provenance,
            )
            if args.prm_scorer_role == "submitted":
                required = (
                    *common,
                    args.source_checkpoint_sha256,
                    args.checkpoint_equivalence_report,
                )
                message = (
                    "formal submitted runs require model/adapter/source hashes, "
                    "checkpoint provenance, and equivalence report"
                )
            else:
                required = (
                    *common,
                    args.checkpoint_provenance_sha256,
                    args.checkpoint_pair_report,
                    args.checkpoint_pair_report_sha256,
                )
                message = (
                    "formal matched runs require model/checkpoint/provenance "
                    "and pair-report hashes"
                )
            if any(not value for value in required):
                raise ValueError(message)


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge_shards(args)
        return

    examples, dataset_fingerprint = load_gsm8k(args.split)
    selected_ids = parse_problem_ids(
        args.problem_ids,
        total=len(examples),
        max_problems=args.max_problems,
    )
    shard_ids = selected_ids[args.shard_id :: args.num_shards]
    if not shard_ids:
        raise ValueError("this shard received zero selected problems")
    if args.formal_run and len(selected_ids) != 1319:
        raise ValueError("formal run selection must contain all 1319 problems")

    from dream.modeling_dream import DreamModel
    from transformers import AutoTokenizer

    LOGGER.info(
        "Loading Dream generator for M=%d shard %d/%d (%d problems)",
        args.top_m,
        args.shard_id,
        args.num_shards,
        len(shard_ids),
    )
    generation_model = DreamModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )
    generation_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    prm, checkpoint_report = load_diffusion_prm(
        checkpoint=args.prm_checkpoint,
        model_path=args.model_path,
        device="cuda",
    )
    if prm.causal or prm.pool_strategy != "mean":
        raise ValueError(
            "formal baseline scorer must be bidirectional mean-pool; "
            f"got causal={prm.causal}, pool={prm.pool_strategy}"
        )

    baseline_config = StableTopMConfig(top_m=args.top_m)
    baseline_config.validate(strict_fixed_protocol=True)
    model_artifact = artifact_descriptor(
        args.model_path,
        sha256=args.model_sha256,
        require_hash=args.formal_run,
        verify_supplied_hash=False,
    )
    checkpoint_artifact = artifact_descriptor(
        checkpoint_report.checkpoint_path,
        sha256=args.checkpoint_sha256,
        require_hash=args.formal_run,
        verify_supplied_hash=args.formal_run,
    )
    submitted_main_gate = None
    matched_prm_gate = None
    provenance_artifact = None
    equivalence_artifact = None
    pair_report_artifact = None
    model_attestation_artifact = None
    if args.formal_run:
        attestation = validate_artifact_attestation(
            args.model_hash_attestation,
            artifact_path=args.model_path,
            expected_sha256=args.model_sha256,
        )
        model_attestation_artifact = attestation["attestation"]
        provenance = _load_json_mapping(
            args.checkpoint_provenance, label="checkpoint provenance"
        )
        provenance_artifact = artifact_descriptor(
            args.checkpoint_provenance,
            sha256=args.checkpoint_provenance_sha256,
            require_hash=True,
            verify_supplied_hash=args.checkpoint_provenance_sha256 is not None,
        )
        if args.prm_scorer_role == "submitted":
            equivalence = _load_json_mapping(
                args.checkpoint_equivalence_report,
                label="checkpoint equivalence report",
            )
            submitted_main_gate = validate_submitted_main_gate(
                checkpoint_report=checkpoint_report.to_dict(),
                adapter_sha256=checkpoint_artifact["sha256"],
                source_checkpoint_sha256=args.source_checkpoint_sha256,
                provenance=provenance,
                equivalence=equivalence,
            )
            equivalence_artifact = artifact_descriptor(
                args.checkpoint_equivalence_report
            )
        else:
            pair_report = _load_json_mapping(
                args.checkpoint_pair_report,
                label="matched checkpoint pair report",
            )
            pair_report_artifact = artifact_descriptor(
                args.checkpoint_pair_report,
                sha256=args.checkpoint_pair_report_sha256,
                require_hash=True,
                verify_supplied_hash=True,
            )
            matched_prm_gate = validate_matched_checkpoint_gate(
                role=args.prm_scorer_role,
                checkpoint_sha256=checkpoint_artifact["sha256"],
                checkpoint_report=checkpoint_report.to_dict(),
                provenance=provenance,
                provenance_path=args.checkpoint_provenance,
                provenance_sha256=provenance_artifact["sha256"],
                pair_report=pair_report,
                pair_report_path=args.checkpoint_pair_report,
                pair_report_sha256=pair_report_artifact["sha256"],
            )

    selected_question_hash = sha256_text(
        json.dumps(
            [examples[index]["question"] for index in selected_ids],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    artifacts = {
        "dream_backbone": model_artifact,
        (
            "submitted_prm"
            if args.prm_scorer_role == "submitted"
            else "matched_prm"
        ): checkpoint_artifact,
    }
    if args.formal_run:
        artifacts["dream_backbone_hash_attestation"] = (
            model_attestation_artifact
        )
        if args.prm_scorer_role == "submitted":
            artifacts["submitted_prm_provenance"] = provenance_artifact
            artifacts["full_to_adapter_equivalence"] = equivalence_artifact
        else:
            artifacts["matched_prm_provenance"] = provenance_artifact
            artifacts["matched_prm_pair_report"] = pair_report_artifact
    manifest = build_protocol_manifest(
        repo_root=RELEASE_ROOT,
        experiment="gsm8k_stable_prm_guided_topm",
        config={
            **vars(args),
            "baseline": baseline_config.to_dict(),
            "shard_problem_ids": shard_ids,
        },
        artifacts=artifacts,
        prompt_template=build_gsm8k_prompt("{QUESTION}"),
        extraction_fn=extract_gsm8k_answer,
        extra={
            "dataset_name": "openai/gsm8k",
            "dataset_subset": "main",
            "dataset_split": args.split,
            "dataset_fingerprint": dataset_fingerprint,
            "selected_question_list_sha256": selected_question_hash,
            "checkpoint_load_report": checkpoint_report.to_dict(),
            "submitted_main_checkpoint_gate": submitted_main_gate,
            "matched_prm_checkpoint_gate": matched_prm_gate,
            "prm_scorer_role": args.prm_scorer_role,
            "fixed_protocol": (
                "K=8; M in {1,2}; T=128; checkpoints=64,128; "
                "temperature=0.5; top_p=1; maskgit_plus; alg_temp=0.5; "
                "gen_length=512; descending-score/lowest-id ties"
            ),
        },
    )
    if args.formal_run and manifest["git"]["dirty"] is not False:
        raise ValueError("formal top-M run requires a clean release worktree")

    destination = Path(args.output_dir) / shard_filename(args, args.shard_id)
    results: list[dict[str, Any]] = []
    completed: set[int] = set()
    if args.resume and destination.is_file():
        with open(destination, encoding="utf-8") as handle:
            existing = json.load(handle)
        results, prior_manifest = _validate_resume_payload(
            existing,
            args=args,
            shard_ids=shard_ids,
            current_manifest=manifest,
        )
        manifest = dict(prior_manifest)
        completed = {
            int(item["global_problem_id"]) for item in results
        }
        LOGGER.info("Resuming with %d completed problems", len(completed))

    started = time.time()
    for sequence_number, global_id in enumerate(shard_ids, start=1):
        if global_id in completed:
            continue
        result = evaluate_one(
            example=examples[global_id],
            tokenizer=tokenizer,
            generation_model=generation_model,
            prm=prm,
            base_seed=args.seed,
            baseline_config=baseline_config,
        )
        results.append(result)
        completed.add(global_id)
        summary = summarize_results(results)
        LOGGER.info(
            "[%d/%d] M=%d pid=%d acc=%.1f%% oracle=%.1f%% time=%.1fs",
            sequence_number,
            len(shard_ids),
            args.top_m,
            global_id,
            100 * summary["accuracy"],
            100 * summary["oracle_at_8"],
            result["wall_time_seconds"],
        )
        if len(results) % args.save_every == 0:
            atomic_write_json(
                destination,
                {
                    "schema_version": 1,
                    "task": "gsm8k",
                    "split": args.split,
                    "method": "stable_prm_guided_topm",
                    "seed": args.seed,
                    "top_m": args.top_m,
                    "shard_id": args.shard_id,
                    "num_shards": args.num_shards,
                    "selected_problem_ids": shard_ids,
                    "manifest": manifest,
                    "summary": summary,
                    "per_problem": sorted(
                        results, key=lambda item: item["global_problem_id"]
                    ),
                    "wall_time_seconds": time.time() - started,
                },
            )

    if len(completed) != len(shard_ids):
        raise AssertionError(
            f"shard completed {len(completed)} of {len(shard_ids)} problems"
        )
    payload = {
        "schema_version": 1,
        "task": "gsm8k",
        "split": args.split,
        "method": "stable_prm_guided_topm",
        "seed": args.seed,
        "top_m": args.top_m,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "selected_problem_ids": shard_ids,
        "manifest": manifest,
        "summary": summarize_results(results),
        "per_problem": sorted(
            results, key=lambda item: item["global_problem_id"]
        ),
        "wall_time_seconds": time.time() - started,
    }
    atomic_write_json(destination, payload)
    LOGGER.info("Saved completed shard to %s", destination)


if __name__ == "__main__":
    main()
