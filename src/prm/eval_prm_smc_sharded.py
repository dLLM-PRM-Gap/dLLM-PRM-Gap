"""Sharded evaluator for the fixed two-stage ESS-tempered SMC protocol.

Formal usage (one process per GPU):

    python -m src.prm.eval_prm_smc_sharded \
        --split test --seed 42 --shard_id 0 --num_shards 16 \
        --model_path ./models/Dream-v0-Instruct-7B \
        --prm_checkpoint ./checkpoints/prm_adapter \
        --output_dir ./results/smc_s42 \
        --formal_run --model_sha256 ... --checkpoint_sha256 ... \
        --source_checkpoint_sha256 ... \
        --checkpoint_provenance ./results/provenance.json \
        --checkpoint_equivalence_report ./results/equivalence.json

Development runs use ``--split train --max_problems 16``.  Stable per-problem
and per-particle seeds make results invariant to ``num_shards``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping

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

from .checkpointing import load_diffusion_prm
from .eval_prm_guided_sharded import (
    build_gsm8k_prompt,
    extract_gsm8k_answer,
    normalize_answer,
)
from .prm_guided_smc_denoise import SMCConfig, prm_guided_smc_sample
from .protocol import (
    artifact_descriptor,
    atomic_write_json,
    build_protocol_manifest,
    sha256_text,
    validate_artifact_attestation,
)


LOGGER = logging.getLogger("eval_prm_smc")
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

NO_ANSWER_KEY = "<NO_EXTRACTED_ANSWER>"
FORMAL_EQUIVALENCE_ATOL = 1e-5
SUBMITTED_MAIN_EXPECTATIONS = {
    "source_checkpoint_sha256": (
        "39b51b274bdbe59f5252b32aaf9eb1b5290d01d55923c16aa7760674d62e0726"
    ),
    "adapter_sha256": (
        "fce88508c25109ef9c6d98676945f463ae47fe45c8e1e8e2d2c47ccc1ee92efe"
    ),
    "checkpoint_step": 2500,
    "max_steps": 3000,
    "trajectory_dir_basename": "gsm8k_combined",
    "seed": 42,
    "causal": False,
    "pool_strategy": "mean",
    "architecture_config": {
        "causal": False,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_r": 16,
        "no_mask_aware": False,
        "no_step_embed": False,
        "pool_strategy": "mean",
        "reward_hidden": 1024,
        "step_embed_dim": 256,
    },
    "summary": {
        "total_steps": 3000,
        "train_samples": 1_276_560,
        "val_samples": 6_000,
        "best_val_acc": 0.8056666666666666,
    },
}


def _load_json_mapping(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    with open(resolved, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _validate_locked_training_config(
    config: Mapping[str, Any], *, label: str
) -> list[str]:
    errors = []
    trajectory_dir = config.get("trajectory_dir")
    trajectory_basename = (
        Path(str(trajectory_dir)).name if trajectory_dir is not None else None
    )
    expected = SUBMITTED_MAIN_EXPECTATIONS
    if trajectory_basename != expected["trajectory_dir_basename"]:
        errors.append(
            f"{label}.trajectory_dir basename={trajectory_basename!r}, "
            f"expected {expected['trajectory_dir_basename']!r}"
        )
    for key in ("max_steps", "seed", "causal", "pool_strategy"):
        if config.get(key) != expected[key]:
            errors.append(
                f"{label}.{key}={config.get(key)!r}, expected {expected[key]!r}"
            )
    for key, expected_value in expected["architecture_config"].items():
        actual = config.get(key)
        if actual != expected_value:
            errors.append(f"{label}.{key}={actual!r}, expected {expected_value!r}")
    return errors


def validate_submitted_main_gate(
    *,
    checkpoint_report: Mapping[str, Any],
    adapter_sha256: str,
    source_checkpoint_sha256: str,
    provenance: Mapping[str, Any],
    equivalence: Mapping[str, Any],
) -> dict[str, Any]:
    """Hard-fail unless this is the submitted combined-corpus PRM adapter."""
    errors = []
    if (
        source_checkpoint_sha256
        != SUBMITTED_MAIN_EXPECTATIONS["source_checkpoint_sha256"]
    ):
        errors.append(
            "source checkpoint SHA256 is not the audited submitted checkpoint"
        )
    if adapter_sha256 != SUBMITTED_MAIN_EXPECTATIONS["adapter_sha256"]:
        errors.append("adapter SHA256 is not the audited submitted checkpoint adapter")
    if checkpoint_report.get("checkpoint_format") != "safetensors_adapter":
        errors.append(
            "formal SMC requires the extracted safetensors adapter, got "
            f"{checkpoint_report.get('checkpoint_format')!r}"
        )
    report_config = checkpoint_report.get("config")
    if not isinstance(report_config, Mapping):
        errors.append("checkpoint load report is missing its training config")
    else:
        errors.extend(
            _validate_locked_training_config(report_config, label="adapter_config")
        )

    if provenance.get("source_checkpoint_sha256") != source_checkpoint_sha256:
        errors.append("provenance source checkpoint SHA256 does not match lock")
    if provenance.get("adapter_sha256") != adapter_sha256:
        errors.append("provenance adapter SHA256 does not match evaluated adapter")
    if (
        provenance.get("checkpoint_step")
        != SUBMITTED_MAIN_EXPECTATIONS["checkpoint_step"]
    ):
        errors.append(
            f"provenance checkpoint_step={provenance.get('checkpoint_step')!r}, "
            f"expected {SUBMITTED_MAIN_EXPECTATIONS['checkpoint_step']}"
        )
    provenance_config = provenance.get("training_config")
    if not isinstance(provenance_config, Mapping):
        errors.append("provenance is missing training_config")
    else:
        errors.extend(
            _validate_locked_training_config(
                provenance_config, label="provenance.training_config"
            )
        )
        if isinstance(report_config, Mapping) and dict(provenance_config) != dict(
            report_config
        ):
            errors.append("adapter config and provenance training_config differ")
    summary = provenance.get("source_summary")
    if not isinstance(summary, Mapping):
        errors.append("provenance is missing source_summary")
    else:
        for key, expected_value in SUBMITTED_MAIN_EXPECTATIONS["summary"].items():
            actual = summary.get(key)
            if isinstance(expected_value, float):
                matches = isinstance(actual, (int, float)) and math.isclose(
                    float(actual), expected_value, rel_tol=0, abs_tol=1e-12
                )
            else:
                matches = actual == expected_value
            if not matches:
                errors.append(
                    f"provenance.source_summary.{key}={actual!r}, "
                    f"expected {expected_value!r}"
                )

    if equivalence.get("all_passed") is not True:
        errors.append("full-to-adapter equivalence report did not pass")
    if equivalence.get("source_checkpoint_sha256") != source_checkpoint_sha256:
        errors.append("equivalence source checkpoint SHA256 does not match lock")
    if equivalence.get("adapter_sha256") != adapter_sha256:
        errors.append("equivalence adapter SHA256 does not match evaluated adapter")
    if int(equivalence.get("num_states", 0)) < 3:
        errors.append("equivalence report must score at least three states")
    max_difference = equivalence.get("max_abs_logit_difference")
    if not isinstance(max_difference, (int, float)) or not math.isfinite(
        float(max_difference)
    ):
        errors.append("equivalence max_abs_logit_difference is not finite")
    elif float(max_difference) > FORMAL_EQUIVALENCE_ATOL:
        errors.append(
            "equivalence max_abs_logit_difference="
            f"{max_difference} exceeds {FORMAL_EQUIVALENCE_ATOL}"
        )
    if equivalence.get("architecture_config_match") is not True:
        errors.append("equivalence architecture configs do not match")
    expected_architecture = SUBMITTED_MAIN_EXPECTATIONS["architecture_config"]
    for key in ("full_architecture_config", "adapter_architecture_config"):
        if equivalence.get(key) != expected_architecture:
            errors.append(f"equivalence {key} is not the submitted architecture")
    adapter_load_report = equivalence.get("adapter_checkpoint_load_report")
    if not isinstance(adapter_load_report, Mapping):
        errors.append("equivalence lacks adapter checkpoint load report")
    else:
        if adapter_load_report.get("checkpoint_format") != "safetensors_adapter":
            errors.append("equivalence adapter load format is not safetensors")
        if isinstance(report_config, Mapping) and adapter_load_report.get(
            "config"
        ) != dict(report_config):
            errors.append("runtime adapter config differs from equivalence load report")
    requested_ratios = (
        equivalence.get("state_protocol", {}).get("mask_ratios_requested")
        if isinstance(equivalence.get("state_protocol"), Mapping)
        else None
    )
    if requested_ratios != [0.95, 0.5, 0.05]:
        errors.append(
            "equivalence states must use requested mask ratios [0.95, 0.5, 0.05]"
        )

    if errors:
        raise ValueError("submitted-main checkpoint gate failed: " + "; ".join(errors))
    return {
        "all_passed": True,
        "expectations": SUBMITTED_MAIN_EXPECTATIONS,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "adapter_sha256": adapter_sha256,
        "max_abs_logit_difference": float(max_difference),
        "formal_equivalence_atol": FORMAL_EQUIVALENCE_ATOL,
    }


def load_gsm8k(split: str) -> tuple[list[dict[str, Any]], str | None]:
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split=split)
    examples = []
    for global_id, item in enumerate(dataset):
        examples.append(
            {
                "global_problem_id": global_id,
                "question": item["question"],
                "gold_answer": normalize_answer(
                    item["answer"].split("####")[-1].strip()
                ),
                "full_gold_solution": item["answer"],
            }
        )
    return examples, getattr(dataset, "_fingerprint", None)


def parse_problem_ids(
    text: str | None,
    *,
    total: int,
    max_problems: int | None,
) -> list[int]:
    if text:
        selected = [int(value.strip()) for value in text.split(",") if value.strip()]
        if len(selected) != len(set(selected)):
            raise ValueError("problem_ids contains duplicates")
        if any(index < 0 or index >= total for index in selected):
            raise ValueError(f"problem_ids must lie in [0, {total})")
    else:
        selected = list(range(total))
    if max_problems is not None:
        if max_problems <= 0:
            raise ValueError("max_problems must be positive")
        selected = selected[:max_problems]
    if not selected:
        raise ValueError("no problems selected")
    return selected


def _answer_key(answer: str | None) -> str:
    return NO_ANSWER_KEY if answer is None else str(answer)


def _entropy(probabilities: Iterable[float]) -> float:
    return -sum(
        probability * math.log2(probability)
        for probability in probabilities
        if probability > 0
    )


def _build_answer_clusters(
    answers: list[str | None],
    weights: list[float],
    *,
    include_invalid: bool = False,
) -> list[dict[str, Any]]:
    clusters: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for particle_id, (answer, weight) in enumerate(zip(answers, weights)):
        if answer is None and not include_invalid:
            continue
        key = _answer_key(answer)
        if key not in clusters:
            clusters[key] = {
                "key": key,
                "answer": answer,
                "particle_indices": [],
                "normalized_weight": 0.0,
                "unweighted_count": 0,
            }
        cluster = clusters[key]
        cluster["particle_indices"].append(particle_id)
        cluster["normalized_weight"] += float(weight)
        cluster["unweighted_count"] += 1
    # Preserve lowest-particle tie breaking while making floating output exact.
    result = list(clusters.values())
    total = sum(cluster["normalized_weight"] for cluster in result)
    if include_invalid and abs(total - 1.0) > 1e-8:
        raise AssertionError(f"cluster weights sum to {total}, expected 1")
    return result


def _select_weighted_cluster_readout(
    valid_clusters: list[dict[str, Any]],
    weights: list[float],
    *,
    gold_answer: str | None,
) -> dict[str, Any]:
    """Apply the fixed valid-answer aggregation and total fallback rule."""
    if valid_clusters:
        weighted_cluster = max(
            valid_clusters, key=lambda cluster: cluster["normalized_weight"]
        )
        return {
            "selected_answer": weighted_cluster["answer"],
            "selected_cluster_key": weighted_cluster["key"],
            "selected_cluster_weight_raw": weighted_cluster["normalized_weight"],
            "selected_cluster_weight_valid_renormalized": weighted_cluster[
                "valid_renormalized_weight"
            ],
            "correct": weighted_cluster["answer"] == gold_answer,
            "fallback_used": False,
            "fallback_particle_id": None,
            "tie_break": "lowest_particle_id",
        }

    # max() keeps the first index on an exact tie.
    fallback_particle_id = max(range(len(weights)), key=lambda index: weights[index])
    return {
        "selected_answer": None,
        "selected_cluster_key": None,
        "selected_cluster_weight_raw": weights[fallback_particle_id],
        "selected_cluster_weight_valid_renormalized": None,
        "correct": False,
        "fallback_used": True,
        "fallback_particle_id": fallback_particle_id,
        "tie_break": "lowest_particle_id",
    }


def evaluate_one(
    *,
    example: Mapping[str, Any],
    tokenizer,
    generation_model,
    prm,
    base_seed: int,
    smc_config: SMCConfig,
) -> dict[str, Any]:
    question = str(example["question"])
    prompt_text = build_gsm8k_prompt(question)
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to("cuda")
    prompt_length = input_ids.shape[-1]
    started = time.time()
    final_states, smc_info = prm_guided_smc_sample(
        generation_model,
        prm,
        input_ids,
        base_seed=base_seed,
        global_problem_id=example["global_problem_id"],
        config=smc_config,
        strict_fixed_protocol=True,
    )
    elapsed = time.time() - started

    terminal = smc_info["terminal"]
    stage1 = smc_info["stage1"]
    resampling = smc_info["resampling"]
    weights = [float(value) for value in terminal["normalized_weights"]]
    ancestors = [int(value) for value in resampling["ancestors"]]
    gold = normalize_answer(example["gold_answer"])

    answers: list[str | None] = []
    candidate_records: list[dict[str, Any]] = []
    for particle_id in range(smc_config.particles):
        generated_ids = (
            final_states[particle_id, prompt_length:].detach().cpu().tolist()
        )
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        answer = normalize_answer(extract_gsm8k_answer(generated_text))
        answers.append(answer)
        ancestor = ancestors[particle_id]
        candidate_records.append(
            {
                "particle_id": particle_id,
                "ancestor": ancestor,
                "stage1_seed": int(stage1["row_seeds"][ancestor]),
                "stage2_seed": int(terminal["row_seeds"][particle_id]),
                "generated_token_ids": generated_ids,
                "generated_text": generated_text,
                "extracted_answer": answer,
                "correct": answer == gold,
                "stage1_parent_raw_prm_logit": float(
                    stage1["raw_prm_logits"][ancestor]
                ),
                "terminal_raw_prm_logit": float(
                    terminal["raw_prm_logits"][particle_id]
                ),
                "normalized_weight": weights[particle_id],
            }
        )

    valid_clusters = _build_answer_clusters(answers, weights)
    raw_clusters = _build_answer_clusters(answers, weights, include_invalid=True)
    invalid_indices = [
        particle_id for particle_id, answer in enumerate(answers) if answer is None
    ]
    invalid_weight_mass = sum(weights[index] for index in invalid_indices)
    valid_weight_mass = 1.0 - invalid_weight_mass
    valid_candidate_count = smc_config.particles - len(invalid_indices)
    for cluster in valid_clusters:
        cluster["valid_renormalized_weight"] = (
            cluster["normalized_weight"] / valid_weight_mass
            if valid_weight_mass > 0
            else None
        )

    weighted_readout = _select_weighted_cluster_readout(
        valid_clusters, weights, gold_answer=gold
    )
    top_particle_index = int(terminal["top_particle_index"])
    top_particle = candidate_records[top_particle_index]
    valid_uniform_cluster_probabilities = (
        [
            cluster["unweighted_count"] / valid_candidate_count
            for cluster in valid_clusters
        ]
        if valid_candidate_count
        else []
    )
    valid_weighted_cluster_probabilities = [
        cluster["valid_renormalized_weight"] for cluster in valid_clusters
    ]
    raw_uniform_cluster_probabilities = [
        cluster["unweighted_count"] / smc_config.particles for cluster in raw_clusters
    ]
    raw_weighted_cluster_probabilities = [
        cluster["normalized_weight"] for cluster in raw_clusters
    ]

    expected_counts = {
        "logical_denoising": smc_config.particles * smc_config.steps,
        "logical_prm_scoring": 2 * smc_config.particles,
        "logical_total": (
            smc_config.particles * smc_config.steps + 2 * smc_config.particles
        ),
        "physical_denoising_forwards": smc_config.steps,
        "physical_prm_forwards": 2,
    }
    if smc_info["call_counts"] != expected_counts:
        raise AssertionError(
            f"call-count mismatch: {smc_info['call_counts']} != {expected_counts}"
        )

    return {
        "global_problem_id": int(example["global_problem_id"]),
        "question": question,
        "gold_answer": gold,
        "prompt_sha256": sha256_text(prompt_text),
        "prompt_length": prompt_length,
        "base_seed": int(base_seed),
        "weighted_cluster_readout": weighted_readout,
        "top_particle_readout": {
            "particle_id": top_particle_index,
            "selected_answer": top_particle["extracted_answer"],
            "raw_terminal_prm_logit": top_particle["terminal_raw_prm_logit"],
            "correct": top_particle["correct"],
        },
        "oracle_correct": any(candidate["correct"] for candidate in candidate_records),
        "valid_candidate_count": valid_candidate_count,
        "invalid_candidate_count": len(invalid_indices),
        "invalid_particle_indices": invalid_indices,
        "invalid_weight_mass": invalid_weight_mass,
        "unique_extracted_answers": len(valid_clusters),
        "valid_unweighted_answer_entropy_bits": _entropy(
            valid_uniform_cluster_probabilities
        ),
        "valid_renormalized_weighted_answer_entropy_bits": _entropy(
            valid_weighted_cluster_probabilities
        ),
        "raw_unweighted_answer_entropy_including_invalid_bits": _entropy(
            raw_uniform_cluster_probabilities
        ),
        "raw_weighted_answer_entropy_including_invalid_bits": _entropy(
            raw_weighted_cluster_probabilities
        ),
        "answer_clusters": valid_clusters,
        "raw_answer_clusters_including_invalid": raw_clusters,
        "candidates": candidate_records,
        "stage1": stage1,
        "resampling": resampling,
        "terminal": terminal,
        "call_counts": smc_info["call_counts"],
        "wall_time_seconds": elapsed,
    }


def summarize_results(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("cannot summarize zero results")
    count = len(results)

    def mean(key_fn) -> float:
        return sum(float(key_fn(item)) for item in results) / count

    call_totals = {
        key: sum(int(item["call_counts"][key]) for item in results)
        for key in results[0]["call_counts"]
    }
    return {
        "num_problems": count,
        "weighted_cluster_accuracy": mean(
            lambda item: item["weighted_cluster_readout"]["correct"]
        ),
        "top_particle_accuracy": mean(
            lambda item: item["top_particle_readout"]["correct"]
        ),
        "oracle_at_8": mean(lambda item: item["oracle_correct"]),
        "mean_unique_extracted_answers": mean(
            lambda item: item["unique_extracted_answers"]
        ),
        "mean_valid_unweighted_answer_entropy_bits": mean(
            lambda item: item["valid_unweighted_answer_entropy_bits"]
        ),
        "mean_valid_renormalized_weighted_answer_entropy_bits": mean(
            lambda item: item["valid_renormalized_weighted_answer_entropy_bits"]
        ),
        "mean_raw_unweighted_answer_entropy_including_invalid_bits": mean(
            lambda item: item["raw_unweighted_answer_entropy_including_invalid_bits"]
        ),
        "mean_raw_weighted_answer_entropy_including_invalid_bits": mean(
            lambda item: item["raw_weighted_answer_entropy_including_invalid_bits"]
        ),
        "mean_invalid_candidate_count": mean(
            lambda item: item["invalid_candidate_count"]
        ),
        "mean_invalid_weight_mass": mean(lambda item: item["invalid_weight_mass"]),
        "weighted_cluster_fallback_rate": mean(
            lambda item: item["weighted_cluster_readout"]["fallback_used"]
        ),
        "mean_step64_ess": mean(lambda item: item["stage1"]["ess"]),
        "resampling_trigger_rate": mean(lambda item: item["resampling"]["triggered"]),
        "mean_unique_ancestor_count": mean(
            lambda item: item["resampling"]["unique_ancestor_count"]
        ),
        "mean_final_lineage_count": mean(
            lambda item: item["terminal"]["final_lineage_count"]
        ),
        "mean_wall_time_seconds": mean(lambda item: item["wall_time_seconds"]),
        "call_count_totals": call_totals,
    }


def shard_filename(args: argparse.Namespace, shard_id: int) -> str:
    return (
        f"smc_gsm8k_{args.split}_K8_s{args.seed}"
        f"_shard{shard_id:03d}-of-{args.num_shards:03d}.json"
    )


def _merge_protocol_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Drop only the two intended shard-varying fields from a run manifest."""
    config = dict(manifest.get("config", {}))
    config.pop("shard_id", None)
    config.pop("shard_problem_ids", None)
    return {
        "schema_version": manifest.get("schema_version"),
        "experiment": manifest.get("experiment"),
        "git": manifest.get("git"),
        "runtime": manifest.get("runtime"),
        "config": config,
        "artifacts": manifest.get("artifacts"),
        "prompt_template_sha256": manifest.get("prompt_template_sha256"),
        "extraction_callable_sha256": manifest.get("extraction_callable_sha256"),
        "stable_seed_spec": manifest.get("stable_seed_spec"),
        "extra": manifest.get("extra"),
    }


def validate_smc_shard_for_merge(
    shard: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    shard_id: int,
    reference_protocol: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Reject incomplete, mixed-protocol, or incorrectly assigned SMC shards."""
    expected_envelope = {
        "schema_version": 1,
        "task": "gsm8k",
        "split": args.split,
        "method": "ess_tempered_smc",
        "seed": args.seed,
        "num_shards": args.num_shards,
        "shard_id": shard_id,
    }
    errors = [
        f"{key}={shard.get(key)!r}, expected {expected!r}"
        for key, expected in expected_envelope.items()
        if shard.get(key) != expected
    ]
    rows = shard.get("per_problem")
    if not isinstance(rows, list) or not rows:
        errors.append("per_problem is empty or not a list")
        rows = []
    row_ids = [int(row["global_problem_id"]) for row in rows]
    selected_ids = [int(value) for value in shard.get("selected_problem_ids", [])]
    if row_ids != selected_ids:
        errors.append("per_problem IDs differ from selected_problem_ids")
    if len(row_ids) != len(set(row_ids)):
        errors.append("shard contains duplicate problem IDs")

    manifest = shard.get("manifest")
    if not isinstance(manifest, Mapping):
        errors.append("shard lacks a protocol manifest")
        protocol_identity: dict[str, Any] = {}
    else:
        config = manifest.get("config", {})
        if (
            not isinstance(config, Mapping)
            or config.get("shard_id") != shard_id
            or config.get("shard_problem_ids") != selected_ids
        ):
            errors.append("manifest shard assignment differs from payload")
        protocol_identity = _merge_protocol_identity(manifest)
        if reference_protocol is not None and protocol_identity != reference_protocol:
            errors.append("shard uses a different code/runtime/protocol identity")

    if rows and shard.get("summary") != summarize_results(rows):
        errors.append("shard summary does not reproduce its per-problem records")

    if args.formal_run:
        expected_ids = list(range(1319))[shard_id :: args.num_shards]
        if selected_ids != expected_ids:
            errors.append("formal shard does not use the canonical strided assignment")
        if isinstance(manifest, Mapping):
            git = manifest.get("git", {})
            gate = manifest.get("extra", {}).get("submitted_main_checkpoint_gate")
            if (
                not git.get("commit")
                or git.get("dirty") is not False
                or not isinstance(gate, Mapping)
                or gate.get("all_passed") is not True
            ):
                errors.append("formal shard lacks clean source/submitted-main gate")
        expected_calls = {
            "logical_denoising": 1024,
            "logical_prm_scoring": 16,
            "logical_total": 1040,
            "physical_denoising_forwards": 128,
            "physical_prm_forwards": 2,
        }
        for row in rows:
            if row.get("call_counts") != expected_calls:
                errors.append(
                    f"problem {row.get('global_problem_id')} call counts are non-canonical"
                )
                break

    if errors:
        raise ValueError(f"invalid SMC shard {shard_id}: " + "; ".join(errors))
    return protocol_identity


def merge_shards(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    results: list[dict[str, Any]] = []
    shard_manifests = []
    reference_protocol = None
    for shard_id in range(args.num_shards):
        path = output_dir / shard_filename(args, shard_id)
        if not path.is_file():
            raise FileNotFoundError(f"missing SMC shard: {path}")
        with open(path, encoding="utf-8") as handle:
            shard = json.load(handle)
        protocol_identity = validate_smc_shard_for_merge(
            shard,
            args=args,
            shard_id=shard_id,
            reference_protocol=reference_protocol,
        )
        if reference_protocol is None:
            reference_protocol = protocol_identity
        results.extend(shard["per_problem"])
        shard_manifests.append(shard["manifest"])

    ids = [int(item["global_problem_id"]) for item in results]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate global_problem_id across shards")
    results.sort(key=lambda item: item["global_problem_id"])
    if args.expected_problems is not None and len(results) != args.expected_problems:
        raise ValueError(
            f"merged {len(results)} problems, expected {args.expected_problems}"
        )
    if args.formal_run:
        expected_ids = list(range(1319))
        if [int(item["global_problem_id"]) for item in results] != expected_ids:
            raise ValueError(
                "formal GSM8K test merge does not cover exactly ids 0..1318"
            )

    merged = {
        "schema_version": 1,
        "task": "gsm8k",
        "split": args.split,
        "method": "ess_tempered_smc",
        "seed": args.seed,
        "num_shards": args.num_shards,
        "summary": summarize_results(results),
        "per_problem": results,
        "shard_manifests": shard_manifests,
        "merge_protocol_identity": reference_protocol,
    }
    destination = output_dir / (f"smc_gsm8k_{args.split}_K8_s{args.seed}_merged.json")
    atomic_write_json(destination, merged)
    LOGGER.info(
        "MERGED %d problems: weighted=%.2f%% top-particle=%.2f%% oracle=%.2f%%",
        len(results),
        100 * merged["summary"]["weighted_cluster_accuracy"],
        100 * merged["summary"]["top_particle_accuracy"],
        100 * merged["summary"]["oracle_at_8"],
    )
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--split", choices=("train", "test"), default="test")
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
    parser.add_argument("--checkpoint_equivalence_report", default=None)
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
    if args.formal_run:
        if args.split != "test":
            raise ValueError("formal SMC runs must use GSM8K test")
        if args.problem_ids is not None or args.max_problems is not None:
            raise ValueError("formal SMC runs must cover the full test set")
        if not args.merge_only and (
            not args.model_sha256
            or not args.model_hash_attestation
            or not args.checkpoint_sha256
            or not getattr(args, "source_checkpoint_sha256", None)
            or not getattr(args, "checkpoint_provenance", None)
            or not getattr(args, "checkpoint_equivalence_report", None)
        ):
            raise ValueError(
                "formal runs require model/adapter/source hashes, checkpoint "
                "provenance, and a full-to-adapter equivalence report"
            )


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
        "Loading Dream generator from %s for shard %d/%d (%d problems)",
        args.model_path,
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
            "formal SMC scorer must be bidirectional mean-pool; "
            f"got causal={prm.causal}, pool={prm.pool_strategy}"
        )

    smc_config = SMCConfig()
    smc_config.validate(strict_fixed_protocol=True)
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
    provenance_artifact = None
    equivalence_artifact = None
    if args.formal_run:
        model_attestation = validate_artifact_attestation(
            args.model_hash_attestation,
            artifact_path=args.model_path,
            expected_sha256=args.model_sha256,
        )
        provenance = _load_json_mapping(
            args.checkpoint_provenance,
            label="checkpoint provenance",
        )
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
        provenance_artifact = artifact_descriptor(args.checkpoint_provenance)
        equivalence_artifact = artifact_descriptor(args.checkpoint_equivalence_report)
    prompt_template = build_gsm8k_prompt("{QUESTION}")
    selected_question_hash = sha256_text(
        json.dumps(
            [examples[index]["question"] for index in selected_ids],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    manifest_artifacts = {
        "dream_backbone": model_artifact,
        "submitted_prm": checkpoint_artifact,
    }
    if args.formal_run:
        manifest_artifacts.update(
            {
                "dream_backbone_hash_attestation": model_attestation["attestation"],
                "submitted_prm_provenance": provenance_artifact,
                "full_to_adapter_equivalence": equivalence_artifact,
            }
        )
    manifest = build_protocol_manifest(
        repo_root=RELEASE_ROOT,
        experiment="gsm8k_ess_tempered_smc",
        config={
            **vars(args),
            "smc": smc_config.to_dict(),
            "shard_problem_ids": shard_ids,
        },
        artifacts=manifest_artifacts,
        prompt_template=prompt_template,
        extraction_fn=extract_gsm8k_answer,
        extra={
            "dataset_name": "openai/gsm8k",
            "dataset_subset": "main",
            "dataset_split": args.split,
            "dataset_fingerprint": dataset_fingerprint,
            "selected_question_list_sha256": selected_question_hash,
            "checkpoint_load_report": checkpoint_report.to_dict(),
            "submitted_main_checkpoint_gate": submitted_main_gate,
            "readouts": {
                "primary": (
                    "weighted valid extracted-answer cluster; invalid mass excluded; "
                    "no-valid fallback is highest-weight particle and counted wrong; "
                    "exact ties use lowest particle id"
                ),
                "secondary": "largest raw terminal PRM logit",
            },
        },
    )

    destination = Path(args.output_dir) / shard_filename(args, args.shard_id)
    results: list[dict[str, Any]] = []
    completed: set[int] = set()
    if args.resume and destination.is_file():
        with open(destination, encoding="utf-8") as handle:
            existing = json.load(handle)
        results = list(existing.get("per_problem", []))
        completed = {int(item["global_problem_id"]) for item in results}
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
            smc_config=smc_config,
        )
        results.append(result)
        completed.add(global_id)
        summary = summarize_results(results)
        LOGGER.info(
            "[%d/%d] pid=%d weighted=%.1f%% top=%.1f%% ESS=%.2f resample=%s time=%.1fs",
            sequence_number,
            len(shard_ids),
            global_id,
            100 * summary["weighted_cluster_accuracy"],
            100 * summary["top_particle_accuracy"],
            result["stage1"]["ess"],
            result["resampling"]["triggered"],
            result["wall_time_seconds"],
        )
        if len(results) % args.save_every == 0:
            atomic_write_json(
                destination,
                {
                    "schema_version": 1,
                    "task": "gsm8k",
                    "split": args.split,
                    "method": "ess_tempered_smc",
                    "seed": args.seed,
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
    final_payload = {
        "schema_version": 1,
        "task": "gsm8k",
        "split": args.split,
        "method": "ess_tempered_smc",
        "seed": args.seed,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "selected_problem_ids": shard_ids,
        "manifest": manifest,
        "summary": summarize_results(results),
        "per_problem": sorted(results, key=lambda item: item["global_problem_id"]),
        "wall_time_seconds": time.time() - started,
    }
    atomic_write_json(destination, final_payload)
    LOGGER.info("Saved complete shard to %s", destination)


if __name__ == "__main__":
    main()
