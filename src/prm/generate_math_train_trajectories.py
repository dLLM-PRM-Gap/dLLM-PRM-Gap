"""Generate pinned MATH-train trajectories for task-specific PRM/ORM fitting.

Formal generation uses all 7,500 Hendrycks MATH train problems, four
independent trajectories per problem, the submitted Dream sampler, and 24
states per trajectory (initial all-MASK state, every sixth post-step state,
and the final post-step state).  MATH500 is loaded only to enforce a zero
normalized-prompt overlap gate before model allocation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


RELEASE_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(RELEASE_ROOT / "src"))

if __package__ in {None, ""}:
    from prm.math_candidate_pool import (
        fixed_denoise_config,
        sampler_manifest_config,
    )
    from prm.math_protocol import (
        MATH500_QUESTION_LIST_SHA256,
        MATH500_REVISION,
        MATH_TRAIN_EXPECTED_PROBLEMS,
        MATH_TRAIN_QUESTION_LIST_SHA256,
        MATH_TRAIN_REVISION,
        PinnedMathVerifier,
        assert_no_problem_overlap,
        audit_problem_overlap,
        extract_last_boxed_answer,
        format_math_prompt,
        load_math500_examples,
        load_math_train_examples,
        math_prompt_template,
    )
    from prm.protocol import (
        artifact_descriptor,
        atomic_write_json,
        build_protocol_manifest,
        derive_seed,
        sha256_file,
        sha256_text,
        validate_artifact_attestation,
    )
    from prm.resume_denoise import (
        DenoiseConfig,
        initialize_denoise_state,
        run_denoise,
    )
else:
    from .math_candidate_pool import (
        fixed_denoise_config,
        sampler_manifest_config,
    )
    from .math_protocol import (
        MATH500_QUESTION_LIST_SHA256,
        MATH500_REVISION,
        MATH_TRAIN_EXPECTED_PROBLEMS,
        MATH_TRAIN_QUESTION_LIST_SHA256,
        MATH_TRAIN_REVISION,
        PinnedMathVerifier,
        assert_no_problem_overlap,
        audit_problem_overlap,
        extract_last_boxed_answer,
        format_math_prompt,
        load_math500_examples,
        load_math_train_examples,
        math_prompt_template,
    )
    from .protocol import (
        artifact_descriptor,
        atomic_write_json,
        build_protocol_manifest,
        derive_seed,
        sha256_file,
        sha256_text,
        validate_artifact_attestation,
    )
    from .resume_denoise import (
        DenoiseConfig,
        initialize_denoise_state,
        run_denoise,
    )


TRAJECTORY_SCHEMA_VERSION = 1
TRAJECTORY_PROTOCOL_ID = "math_train_chat_binary_4x24_maskgitplus_v1"
TRAJECTORY_SEED_STAGE = "math_train_trajectory_v1"
FORMAL_BASE_SEED = 42
FORMAL_TRAJECTORIES_PER_PROBLEM = 4
FORMAL_TRAJECTORY_BATCH_SIZE = 4
FORMAL_SNAPSHOT_EVERY = 6
FORMAL_VAL_FRACTION = 0.1
FORMAL_SPLIT_SEED = 42


LOGGER = logging.getLogger("generate_math_train_trajectories")
if not LOGGER.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)


def atomic_torch_save(payload: Any, destination: str | os.PathLike[str]) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def trajectory_seed(
    base_seed: int,
    global_problem_id: int | str,
    trajectory_id: int,
) -> int:
    return derive_seed(
        int(base_seed),
        global_problem_id,
        int(trajectory_id),
        TRAJECTORY_SEED_STAGE,
    )


def capture_steps(
    config: DenoiseConfig,
    *,
    snapshot_every: int,
) -> list[int]:
    if snapshot_every <= 0:
        raise ValueError("snapshot_every must be positive")
    selected = list(range(0, config.steps, snapshot_every))
    final_step = config.steps - 1
    if final_step not in selected:
        selected.append(final_step)
    return sorted(selected)


def expected_snapshot_steps(
    config: DenoiseConfig,
    *,
    snapshot_every: int,
) -> list[int]:
    return [-1, *capture_steps(config, snapshot_every=snapshot_every)]


@torch.no_grad()
def generate_problem_trajectories(
    *,
    model,
    tokenizer,
    prompt_input_ids: torch.LongTensor,
    example: Mapping[str, Any],
    base_seed: int,
    trajectory_ids: Sequence[int],
    trajectory_batch_size: int,
    snapshot_every: int,
    verifier: PinnedMathVerifier,
    denoise_config: DenoiseConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], float]:
    """Generate selected trajectories with batch/order-invariant row RNG."""
    config = denoise_config or fixed_denoise_config()
    config.validate()
    if prompt_input_ids.ndim != 2 or prompt_input_ids.shape[0] != 1:
        raise ValueError("prompt_input_ids must have shape [1, prompt_length]")
    selected = [int(value) for value in trajectory_ids]
    if len(selected) != len(set(selected)) or any(value < 0 for value in selected):
        raise ValueError("trajectory_ids must be distinct non-negative integers")
    if trajectory_batch_size <= 0:
        raise ValueError("trajectory_batch_size must be positive")
    post_steps = capture_steps(config, snapshot_every=snapshot_every)
    all_steps = [-1, *post_steps]
    prompt_length = int(prompt_input_ids.shape[-1])
    prompt_token_ids = prompt_input_ids[0].detach().cpu().to(torch.int32)
    records: list[dict[str, Any]] = []
    logical_forwards = 0
    physical_forwards = 0
    started = time.time()

    for start in range(0, len(selected), trajectory_batch_size):
        batch_ids = selected[start : start + trajectory_batch_size]
        row_seeds = [
            trajectory_seed(
                base_seed,
                example["global_problem_id"],
                trajectory_id,
            )
            for trajectory_id in batch_ids
        ]
        prompts = prompt_input_ids.expand(len(batch_ids), -1).clone()
        initial = initialize_denoise_state(prompts, config, row_seeds)
        completed, captures = run_denoise(
            model,
            initial,
            config,
            capture_after_steps=post_steps,
        )
        if completed.logical_denoising_calls != len(batch_ids) * config.steps:
            raise AssertionError("unexpected logical denoising call count")
        if completed.physical_forward_calls != config.steps:
            raise AssertionError("unexpected physical denoising call count")
        logical_forwards += completed.logical_denoising_calls
        physical_forwards += completed.physical_forward_calls

        for row, (trajectory_id, seed) in enumerate(zip(batch_ids, row_seeds)):
            generation_states = [
                initial.x[row, prompt_length:].detach().cpu().to(torch.int32)
            ]
            generation_states.extend(
                captures[step].x[row, prompt_length:].detach().cpu().to(torch.int32)
                for step in post_steps
            )
            snapshots = torch.stack(generation_states)
            mask_ratios = (
                snapshots.eq(config.mask_token_id).float().mean(dim=-1).tolist()
            )
            generated = completed.x[row, prompt_length:].detach().cpu().to(torch.long)
            remaining_masks = int(generated.eq(config.mask_token_id).sum().item())
            if remaining_masks:
                raise AssertionError(
                    f"trajectory {trajectory_id} ended with "
                    f"{remaining_masks} MASK tokens"
                )
            generated_token_ids = generated.tolist()
            text = tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=True,
            )
            grade = verifier.grade_output(text, str(example["gold_answer"]))
            extraction = verifier.extract_answer(text)
            records.append(
                {
                    "schema_version": TRAJECTORY_SCHEMA_VERSION,
                    "protocol_id": TRAJECTORY_PROTOCOL_ID,
                    "task": "math_train",
                    "problem_id": str(example["global_problem_id"]),
                    "global_problem_id": str(example["global_problem_id"]),
                    "problem_ordinal": int(example["problem_ordinal"]),
                    "problem_sha256": str(example["problem_sha256"]),
                    "trajectory_id": int(trajectory_id),
                    "seed": int(seed),
                    "prompt_ids": prompt_token_ids.clone(),
                    "prompt_sha256": sha256_text(
                        json.dumps(
                            prompt_token_ids.tolist(),
                            separators=(",", ":"),
                        )
                    ),
                    "gen_snapshots": snapshots,
                    "snapshot_steps": list(all_steps),
                    "mask_ratios": [float(value) for value in mask_ratios],
                    "generated_token_ids": generated_token_ids,
                    "final_text": text,
                    "answer_extracted": extraction.answer,
                    "extraction": extraction.to_dict(),
                    "answer_target": str(example["gold_answer"]),
                    "grade": grade.to_dict(),
                    "is_correct": grade.correct,
                    "call_counts": {
                        "logical_denoising": config.steps,
                    },
                }
            )
    records.sort(key=lambda item: int(item["trajectory_id"]))
    return (
        records,
        {
            "new_trajectories": len(records),
            "logical_denoising": logical_forwards,
            "physical_denoising_forwards": physical_forwards,
        },
        time.time() - started,
    )


def validate_trajectory_record(
    record: Mapping[str, Any],
    *,
    base_seed: int,
    num_trajectories: int,
    snapshot_every: int,
    denoise_config: DenoiseConfig | None = None,
) -> None:
    config = denoise_config or fixed_denoise_config()
    if record.get("task") != "math_train":
        raise ValueError("trajectory task is not math_train")
    if record.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError("trajectory schema mismatch")
    if record.get("protocol_id") != TRAJECTORY_PROTOCOL_ID:
        raise ValueError("trajectory protocol mismatch")
    if str(record.get("problem_id")) != str(record.get("global_problem_id")):
        raise ValueError("trajectory problem identity fields differ")
    trajectory_id = int(record["trajectory_id"])
    if not (0 <= trajectory_id < num_trajectories):
        raise ValueError("trajectory_id outside canonical range")
    expected_seed = trajectory_seed(
        base_seed,
        record["global_problem_id"],
        trajectory_id,
    )
    if int(record["seed"]) != expected_seed:
        raise ValueError("trajectory seed mismatch")
    expected_steps = expected_snapshot_steps(
        config,
        snapshot_every=snapshot_every,
    )
    if [int(value) for value in record["snapshot_steps"]] != expected_steps:
        raise ValueError("trajectory snapshot schedule mismatch")
    snapshots = record.get("gen_snapshots")
    if not torch.is_tensor(snapshots):
        raise TypeError("gen_snapshots must be a tensor")
    if tuple(snapshots.shape) != (len(expected_steps), config.gen_length):
        raise ValueError(f"unexpected gen_snapshots shape {tuple(snapshots.shape)}")
    if len(record.get("mask_ratios", [])) != len(expected_steps):
        raise ValueError("mask_ratios length mismatch")
    if len(record.get("generated_token_ids", [])) != config.gen_length:
        raise ValueError("generated token length mismatch")
    if record.get("prompt_ids") is None:
        raise ValueError("trajectory prompt_ids must be a nonempty vector")
    prompt_ids = torch.as_tensor(record["prompt_ids"])
    if prompt_ids.ndim != 1 or prompt_ids.numel() == 0:
        raise ValueError("trajectory prompt_ids must be a nonempty vector")
    prompt_digest = sha256_text(
        json.dumps(
            prompt_ids.detach().cpu().tolist(),
            separators=(",", ":"),
        )
    )
    if record.get("prompt_sha256") != prompt_digest:
        raise ValueError("trajectory prompt token hash mismatch")
    if record.get("call_counts") != {"logical_denoising": config.steps}:
        raise ValueError("trajectory call count mismatch")
    grade = record.get("grade")
    if not isinstance(grade, Mapping):
        raise TypeError("trajectory grade must be a mapping")
    if bool(record.get("is_correct")) != bool(grade.get("correct")):
        raise ValueError("trajectory label differs from raw-output grade")
    extraction = record.get("extraction")
    if not isinstance(extraction, Mapping):
        raise TypeError("trajectory extraction must be a mapping")
    if record.get("answer_extracted") != extraction.get("answer"):
        raise ValueError("trajectory extraction record mismatch")


def validate_trajectory_source_identity(
    record: Mapping[str, Any],
    example: Mapping[str, Any],
) -> None:
    """Bind a trajectory row to the immutable pinned train example."""
    expected = {
        "problem_id": str(example["global_problem_id"]),
        "global_problem_id": str(example["global_problem_id"]),
        "problem_ordinal": int(example["problem_ordinal"]),
        "problem_sha256": str(example["problem_sha256"]),
        "answer_target": str(example["gold_answer"]),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"trajectory canonical {key} mismatch")


def shard_filename(args: argparse.Namespace, shard_id: int) -> str:
    return f"shard_{shard_id:03d}-of-{args.num_shards:03d}.pt"


def _manifest_signature(manifest: Mapping[str, Any]) -> dict[str, Any]:
    extra = manifest.get("extra", {})
    return {
        "git_commit": manifest.get("git", {}).get("commit"),
        "git_dirty": manifest.get("git", {}).get("dirty"),
        "dream_backbone": manifest.get("artifacts", {}).get("dream_backbone"),
        "sampler": manifest.get("config", {}).get("sampler"),
        "trajectory_batch_size": manifest.get("config", {}).get(
            "trajectory_batch_size"
        ),
        "snapshot_every": manifest.get("config", {}).get("snapshot_every"),
        "num_trajectories": manifest.get("config", {}).get("num_trajectories"),
        "prompt_template_sha256": manifest.get("prompt_template_sha256"),
        "extraction_callable_sha256": manifest.get("extraction_callable_sha256"),
        "math_train_dataset": extra.get("math_train_dataset"),
        "math500_dataset": extra.get("math500_dataset"),
        "grader": extra.get("math_grader"),
        "overlap_passed": extra.get("train_eval_overlap", {}).get("passed"),
    }


def make_shard_payload(
    *,
    args: argparse.Namespace,
    shard_ordinals: Sequence[int],
    manifest: Mapping[str, Any],
    trajectories: Sequence[Mapping[str, Any]],
    elapsed: float,
    complete: bool,
) -> dict[str, Any]:
    ordered = sorted(
        trajectories,
        key=lambda item: (
            int(item["problem_ordinal"]),
            int(item["trajectory_id"]),
        ),
    )
    return {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "task": "math_train",
        "protocol_id": TRAJECTORY_PROTOCOL_ID,
        "base_seed": int(args.base_seed),
        "num_trajectories": int(args.num_trajectories),
        "snapshot_every": int(args.snapshot_every),
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "selected_problem_ordinals": list(shard_ordinals),
        "complete": bool(complete),
        "manifest": dict(manifest),
        "wall_time_seconds": float(elapsed),
        "trajectories": ordered,
    }


def validate_shard_payload(
    payload: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    expected_shard_id: int,
    shard_ordinals: Sequence[int] | None = None,
    canonical_by_ordinal: Mapping[int, Mapping[str, Any]] | None = None,
) -> None:
    expected = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "task": "math_train",
        "protocol_id": TRAJECTORY_PROTOCOL_ID,
        "base_seed": int(args.base_seed),
        "num_trajectories": int(args.num_trajectories),
        "snapshot_every": int(args.snapshot_every),
        "shard_id": int(expected_shard_id),
        "num_shards": int(args.num_shards),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"trajectory shard {key} mismatch: {payload.get(key)!r} != {value!r}"
            )
    if shard_ordinals is not None and [
        int(item) for item in payload.get("selected_problem_ordinals", [])
    ] != list(shard_ordinals):
        raise ValueError("trajectory shard assignment mismatch")
    trajectories = payload.get("trajectories")
    if not isinstance(trajectories, list):
        raise TypeError("trajectory shard trajectories must be a list")
    seen_pairs: set[tuple[str, int]] = set()
    for record in trajectories:
        pair = (
            str(record["global_problem_id"]),
            int(record["trajectory_id"]),
        )
        if pair in seen_pairs:
            raise ValueError(f"duplicate trajectory pair: {pair}")
        seen_pairs.add(pair)
        validate_trajectory_record(
            record,
            base_seed=args.base_seed,
            num_trajectories=args.num_trajectories,
            snapshot_every=args.snapshot_every,
        )
        ordinal = int(record["problem_ordinal"])
        if canonical_by_ordinal is not None:
            if ordinal not in canonical_by_ordinal:
                raise ValueError("trajectory ordinal is absent from canonical train")
            validate_trajectory_source_identity(
                record,
                canonical_by_ordinal[ordinal],
            )
    if shard_ordinals is not None:
        actual_record_ordinals = sorted(
            int(record["problem_ordinal"]) for record in trajectories
        )
        allowed = set(int(value) for value in shard_ordinals)
        if not set(actual_record_ordinals).issubset(allowed):
            raise ValueError("trajectory records differ from shard assignment")
        if payload.get("complete") is True:
            expected_record_ordinals = [
                ordinal
                for ordinal in shard_ordinals
                for _ in range(args.num_trajectories)
            ]
            if actual_record_ordinals != sorted(expected_record_ordinals):
                raise ValueError("complete trajectory shard misses assigned records")


def freeze_problem_split(
    problem_ids: Sequence[str],
    *,
    seed: int = FORMAL_SPLIT_SEED,
    val_fraction: float = FORMAL_VAL_FRACTION,
) -> dict[str, Any]:
    if not (0 < val_fraction < 1):
        raise ValueError("val_fraction must be in (0, 1)")
    ordered = sorted(str(value) for value in problem_ids)
    if len(ordered) != len(set(ordered)):
        raise ValueError("problem_ids contains duplicates")
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    num_validation = max(1, int(len(shuffled) * val_fraction))
    validation = sorted(shuffled[:num_validation])
    fitting = sorted(shuffled[num_validation:])
    return {
        "schema_version": 1,
        "split_protocol": "sorted_ids_python_random_shuffle_v1",
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "num_fitting_problems": len(fitting),
        "num_validation_problems": len(validation),
        "fitting_problem_ids": fitting,
        "validation_problem_ids": validation,
        "fitting_problem_ids_sha256": sha256_text(
            json.dumps(fitting, separators=(",", ":"))
        ),
        "validation_problem_ids_sha256": sha256_text(
            json.dumps(validation, separators=(",", ":"))
        ),
    }


def merge_shards(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    canonical_by_ordinal: dict[int, dict[str, Any]] | None = None
    canonical_train_metadata: Mapping[str, Any] | None = None
    canonical_evaluation_metadata: Mapping[str, Any] | None = None
    if args.formal_run:
        canonical_examples, canonical_train_metadata = load_math_train_examples(
            revision=args.math_train_revision,
            formal_run=True,
            cache_dir=args.cache_dir,
        )
        evaluation_examples, canonical_evaluation_metadata = load_math500_examples(
            revision=args.math500_revision,
            formal_run=True,
            cache_dir=args.cache_dir,
        )
        assert_no_problem_overlap(
            audit_problem_overlap(canonical_examples, evaluation_examples)
        )
        canonical_by_ordinal = {
            int(item["problem_ordinal"]): dict(item) for item in canonical_examples
        }
    all_trajectories: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, int]] = set()
    reference_signature: dict[str, Any] | None = None
    for shard_id in range(args.num_shards):
        path = output_dir / shard_filename(args, shard_id)
        if not path.is_file():
            raise FileNotFoundError(f"missing trajectory shard: {path}")
        payload = torch.load(path, weights_only=False, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise TypeError(f"trajectory shard is not a mapping: {path}")
        validate_shard_payload(
            payload,
            args=args,
            expected_shard_id=shard_id,
            shard_ordinals=list(range(MATH_TRAIN_EXPECTED_PROBLEMS))[
                shard_id :: args.num_shards
            ]
            if args.formal_run
            else None,
            canonical_by_ordinal=canonical_by_ordinal,
        )
        if payload.get("complete") is not True:
            raise ValueError(f"trajectory shard {shard_id} is incomplete")
        manifest = payload.get("manifest")
        if not isinstance(manifest, Mapping):
            raise TypeError("trajectory shard manifest must be a mapping")
        signature = _manifest_signature(manifest)
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise ValueError("trajectory shards use different protocols")
        if args.formal_run:
            if signature["git_dirty"] is not False or not signature["git_commit"]:
                raise ValueError("formal trajectory shard has dirty source")
            model_artifact = signature.get("dream_backbone")
            if not isinstance(model_artifact, Mapping) or not model_artifact.get(
                "sha256"
            ):
                raise ValueError("formal trajectory shard lacks Dream hash")
            if signature["overlap_passed"] is not True:
                raise ValueError("formal trajectory shard failed overlap audit")
            if signature["sampler"] != sampler_manifest_config():
                raise ValueError("formal trajectory shard sampler mismatch")
            if signature["trajectory_batch_size"] != FORMAL_TRAJECTORY_BATCH_SIZE:
                raise ValueError("formal trajectory shard batch size mismatch")
            if (
                not signature["prompt_template_sha256"]
                or not (signature["extraction_callable_sha256"])
            ):
                raise ValueError("formal trajectory shard lacks code hashes")
            manifest_config = manifest.get("config", {})
            expected_assignment = list(range(MATH_TRAIN_EXPECTED_PROBLEMS))[
                shard_id :: args.num_shards
            ]
            if (
                manifest_config.get("shard_id") != shard_id
                or manifest_config.get("num_shards") != args.num_shards
                or manifest_config.get("shard_problem_ordinals") != expected_assignment
            ):
                raise ValueError("formal trajectory manifest shard metadata mismatch")
            if canonical_train_metadata is None or (
                canonical_evaluation_metadata is None
            ):
                raise AssertionError("formal canonical dataset metadata missing")
            extra = manifest.get("extra", {})
            if extra.get("math_train_dataset") != dict(canonical_train_metadata):
                raise ValueError("formal trajectory train dataset identity mismatch")
            if extra.get("math500_dataset") != dict(canonical_evaluation_metadata):
                raise ValueError("formal trajectory MATH500 identity mismatch")
            if (
                extra["math_train_dataset"].get("question_list_sha256")
                != MATH_TRAIN_QUESTION_LIST_SHA256
                or extra["math500_dataset"].get("question_list_sha256")
                != MATH500_QUESTION_LIST_SHA256
            ):
                raise ValueError("formal trajectory question-list lock mismatch")
        manifests.append(dict(manifest))
        for raw in payload["trajectories"]:
            record = dict(raw)
            pair = (
                str(record["global_problem_id"]),
                int(record["trajectory_id"]),
            )
            if pair in seen_pairs:
                raise ValueError(f"duplicate merged trajectory: {pair}")
            seen_pairs.add(pair)
            all_trajectories.append(record)

    all_trajectories.sort(
        key=lambda item: (
            int(item["problem_ordinal"]),
            int(item["trajectory_id"]),
        )
    )
    by_problem: dict[str, set[int]] = {}
    ordinal_by_problem: dict[str, int] = {}
    for record in all_trajectories:
        problem_id = str(record["global_problem_id"])
        by_problem.setdefault(problem_id, set()).add(int(record["trajectory_id"]))
        ordinal_by_problem[problem_id] = int(record["problem_ordinal"])
    expected_ids = set(range(args.num_trajectories))
    incomplete = {
        problem_id: sorted(ids)
        for problem_id, ids in by_problem.items()
        if ids != expected_ids
    }
    if incomplete:
        first = list(incomplete.items())[:5]
        raise ValueError(f"incomplete trajectories per problem: {first}")
    if args.expected_problems is not None and len(by_problem) != (
        args.expected_problems
    ):
        raise ValueError(
            f"merged {len(by_problem)} problems; expected {args.expected_problems}"
        )
    if args.formal_run and len(by_problem) != MATH_TRAIN_EXPECTED_PROBLEMS:
        raise ValueError("formal merge must contain all 7,500 MATH train problems")
    observed_ordinals = sorted(ordinal_by_problem.values())
    if observed_ordinals != list(range(len(by_problem))):
        raise ValueError("merged problem ordinals are not contiguous")

    merged_path = output_dir / "trajectories_all.pt"
    atomic_torch_save(all_trajectories, merged_path)
    merged_sha256 = sha256_file(merged_path)
    split = freeze_problem_split(sorted(by_problem))
    atomic_write_json(output_dir / "problem_split.json", split)
    num_correct = sum(bool(record["is_correct"]) for record in all_trajectories)
    grade_status_counts: dict[str, int] = {}
    for record in all_trajectories:
        status = str(record["grade"]["status"])
        grade_status_counts[status] = grade_status_counts.get(status, 0) + 1
    merged_manifest = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "task": "math_train",
        "protocol_id": TRAJECTORY_PROTOCOL_ID,
        "complete": True,
        "num_shards": args.num_shards,
        "num_problems": len(by_problem),
        "num_trajectories": len(all_trajectories),
        "trajectories_per_problem": args.num_trajectories,
        "correct": num_correct,
        "accuracy": (num_correct / len(all_trajectories) if all_trajectories else 0.0),
        "grade_status_counts": grade_status_counts,
        "trajectory_artifact": {
            "path": str(merged_path.resolve()),
            "sha256": merged_sha256,
            "sha256_scheme": "file-bytes",
        },
        "problem_split": split,
        "shard_manifests": manifests,
    }
    atomic_write_json(output_dir / "manifest_merged.json", merged_manifest)
    LOGGER.info(
        "MERGED problems=%d trajectories=%d correct=%d (%.2f%%) sha256=%s",
        len(by_problem),
        len(all_trajectories),
        num_correct,
        100 * merged_manifest["accuracy"],
        merged_sha256,
    )
    return merged_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--model_sha256", default=None)
    parser.add_argument("--model_hash_attestation", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_seed", type=int, default=FORMAL_BASE_SEED)
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=FORMAL_TRAJECTORIES_PER_PROBLEM,
    )
    parser.add_argument("--trajectory_batch_size", type=int, default=4)
    parser.add_argument(
        "--snapshot_every",
        type=int,
        default=FORMAL_SNAPSHOT_EVERY,
    )
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--problem_ordinals", default=None)
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge_only", action="store_true")
    parser.add_argument("--expected_problems", type=int, default=None)
    parser.add_argument("--math_train_revision", default=MATH_TRAIN_REVISION)
    parser.add_argument("--math500_revision", default=MATH500_REVISION)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--formal_run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("shard_id must lie in [0, num_shards)")
    if args.num_trajectories <= 0 or args.trajectory_batch_size <= 0:
        raise ValueError("trajectory counts and batch size must be positive")
    if args.snapshot_every <= 0 or args.save_every <= 0:
        raise ValueError("snapshot_every and save_every must be positive")
    if not args.merge_only and not args.model_path:
        raise ValueError("generation requires --model_path")
    if args.formal_run:
        expected = {
            "base_seed": FORMAL_BASE_SEED,
            "num_trajectories": FORMAL_TRAJECTORIES_PER_PROBLEM,
            "trajectory_batch_size": FORMAL_TRAJECTORY_BATCH_SIZE,
            "snapshot_every": FORMAL_SNAPSHOT_EVERY,
        }
        mismatches = {
            key: {"expected": value, "actual": getattr(args, key)}
            for key, value in expected.items()
            if getattr(args, key) != value
        }
        if mismatches:
            raise ValueError(
                "formal trajectory configuration mismatch: "
                + json.dumps(mismatches, sort_keys=True)
            )
        if args.problem_ordinals is not None or args.max_problems is not None:
            raise ValueError("formal generation must cover all MATH train problems")
        if args.num_shards > MATH_TRAIN_EXPECTED_PROBLEMS:
            raise ValueError("formal num_shards cannot exceed 7,500")
        if not args.merge_only and (
            not args.model_sha256 or not args.model_hash_attestation
        ):
            raise ValueError(
                "formal generation requires --model_sha256 and --model_hash_attestation"
            )


def parse_problem_ordinals(
    text: str | None,
    *,
    total: int,
    max_problems: int | None,
) -> list[int]:
    if text:
        selected = [int(item.strip()) for item in text.split(",") if item.strip()]
        if len(selected) != len(set(selected)):
            raise ValueError("problem_ordinals contains duplicates")
        if any(item < 0 or item >= total for item in selected):
            raise ValueError(f"problem_ordinals must lie in [0, {total})")
    else:
        selected = list(range(total))
    if max_problems is not None:
        if max_problems <= 0:
            raise ValueError("max_problems must be positive")
        selected = selected[:max_problems]
    if not selected:
        raise ValueError("no problems selected")
    return selected


def _build_generation_manifest(
    *,
    args: argparse.Namespace,
    shard_ordinals: Sequence[int],
    train_metadata: Mapping[str, Any],
    evaluation_metadata: Mapping[str, Any],
    overlap_report: Mapping[str, Any],
    verifier: PinnedMathVerifier,
    model_artifact: Mapping[str, Any],
    attestation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    artifacts: dict[str, Mapping[str, Any]] = {
        "dream_backbone": model_artifact,
    }
    if attestation is not None:
        artifacts["dream_backbone_hash_attestation"] = attestation
    return build_protocol_manifest(
        repo_root=RELEASE_ROOT,
        experiment="math_train_trajectory_generation",
        config={
            "protocol_id": TRAJECTORY_PROTOCOL_ID,
            "base_seed": args.base_seed,
            "num_trajectories": args.num_trajectories,
            "trajectory_batch_size": args.trajectory_batch_size,
            "snapshot_every": args.snapshot_every,
            "snapshot_steps": expected_snapshot_steps(
                fixed_denoise_config(),
                snapshot_every=args.snapshot_every,
            ),
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "shard_problem_ordinals": list(shard_ordinals),
            "sampler": sampler_manifest_config(),
        },
        artifacts=artifacts,
        prompt_template=math_prompt_template(),
        extraction_fn=extract_last_boxed_answer,
        dependency_names=(
            "torch",
            "transformers",
            "datasets",
            "peft",
            "safetensors",
            "numpy",
            "math-verify",
            "antlr4-python3-runtime",
        ),
        extra={
            "math_train_dataset": dict(train_metadata),
            "math500_dataset": dict(evaluation_metadata),
            "train_eval_overlap": dict(overlap_report),
            "math_grader": verifier.manifest(),
            "prompt_mode": "dream_chat_template",
            "trajectory_seed_key": (
                "(base_seed, global_problem_id, trajectory_id, "
                "'math_train_trajectory_v1')"
            ),
            "label_contract": "raw_final_text_math_verify_binary",
        },
    )


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge_shards(args)
        return

    # Dataset, overlap, package, artifact, and clean-source gates all run before
    # the 7B model is allocated.
    examples, train_metadata = load_math_train_examples(
        revision=args.math_train_revision,
        formal_run=args.formal_run,
        cache_dir=args.cache_dir,
    )
    evaluation_examples, evaluation_metadata = load_math500_examples(
        revision=args.math500_revision,
        formal_run=args.formal_run,
        cache_dir=args.cache_dir,
    )
    overlap_report = audit_problem_overlap(examples, evaluation_examples)
    assert_no_problem_overlap(overlap_report)
    verifier = PinnedMathVerifier(formal_run=args.formal_run)
    selected_ordinals = parse_problem_ordinals(
        args.problem_ordinals,
        total=len(examples),
        max_problems=args.max_problems,
    )
    if args.formal_run and selected_ordinals != list(
        range(MATH_TRAIN_EXPECTED_PROBLEMS)
    ):
        raise ValueError("formal selection must contain all MATH train problems")
    shard_ordinals = selected_ordinals[args.shard_id :: args.num_shards]
    if not shard_ordinals:
        raise ValueError("this shard received zero selected problems")

    model_artifact = artifact_descriptor(
        args.model_path,
        sha256=args.model_sha256,
        require_hash=args.formal_run,
        verify_supplied_hash=False,
    )
    attestation = None
    if args.formal_run:
        attestation_payload = validate_artifact_attestation(
            args.model_hash_attestation,
            artifact_path=args.model_path,
            expected_sha256=args.model_sha256,
        )
        attestation = attestation_payload["attestation"]
    manifest = _build_generation_manifest(
        args=args,
        shard_ordinals=shard_ordinals,
        train_metadata=train_metadata,
        evaluation_metadata=evaluation_metadata,
        overlap_report=overlap_report,
        verifier=verifier,
        model_artifact=model_artifact,
        attestation=attestation,
    )
    if args.formal_run:
        git = manifest.get("git", {})
        if not git.get("commit") or git.get("dirty") is not False:
            raise ValueError(
                "formal generation requires a clean committed release tree"
            )

    destination = output_dir / shard_filename(args, args.shard_id)
    trajectories: list[dict[str, Any]] = []
    completed_ordinals: set[int] = set()
    prior_elapsed = 0.0
    if destination.exists():
        if not args.resume:
            raise FileExistsError(
                f"{destination} exists; pass --resume or use a new directory"
            )
        payload = torch.load(
            destination,
            weights_only=False,
            map_location="cpu",
        )
        if not isinstance(payload, Mapping):
            raise TypeError("resume trajectory shard is not a mapping")
        validate_shard_payload(
            payload,
            args=args,
            expected_shard_id=args.shard_id,
            shard_ordinals=shard_ordinals,
            canonical_by_ordinal={
                int(item["problem_ordinal"]): item for item in examples
            }
            if args.formal_run
            else None,
        )
        if _manifest_signature(payload["manifest"]) != _manifest_signature(manifest):
            raise ValueError("resume protocol/artifact signature mismatch")
        manifest = dict(payload["manifest"])
        trajectories = [dict(item) for item in payload.get("trajectories", [])]
        by_ordinal: dict[int, set[int]] = {}
        for record in trajectories:
            by_ordinal.setdefault(int(record["problem_ordinal"]), set()).add(
                int(record["trajectory_id"])
            )
        expected_ids = set(range(args.num_trajectories))
        partial = {
            ordinal: ids for ordinal, ids in by_ordinal.items() if ids != expected_ids
        }
        if partial:
            raise ValueError(
                "resume shard contains a partially saved problem: "
                f"{list(partial.items())[:5]}"
            )
        completed_ordinals = set(by_ordinal)
        prior_elapsed = float(payload.get("wall_time_seconds", 0.0))
        LOGGER.info("Resuming with %d completed problems", len(completed_ordinals))

    if completed_ordinals == set(shard_ordinals):
        atomic_torch_save(
            make_shard_payload(
                args=args,
                shard_ordinals=shard_ordinals,
                manifest=manifest,
                trajectories=trajectories,
                elapsed=prior_elapsed,
                complete=True,
            ),
            destination,
        )
        LOGGER.info("Trajectory shard is already complete: %s", destination)
        return

    from dream.modeling_dream import DreamModel
    from transformers import AutoTokenizer

    LOGGER.info(
        "Loading Dream from %s for shard=%d/%d (%d problems, %d trajectories each)",
        args.model_path,
        args.shard_id,
        args.num_shards,
        len(shard_ordinals),
        args.num_trajectories,
    )
    model = DreamModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    config = fixed_denoise_config()
    started = time.time()
    newly_completed = 0
    for ordinal in shard_ordinals:
        if ordinal in completed_ordinals:
            continue
        example = examples[ordinal]
        if int(example["problem_ordinal"]) != ordinal:
            raise AssertionError("MATH train problem ordinal drift")
        prompt_text = format_math_prompt(example["question"], tokenizer)
        prompt_input_ids = tokenizer(
            prompt_text,
            return_tensors="pt",
        )["input_ids"].to("cuda")
        new_records, execution, generation_seconds = generate_problem_trajectories(
            model=model,
            tokenizer=tokenizer,
            prompt_input_ids=prompt_input_ids,
            example=example,
            base_seed=args.base_seed,
            trajectory_ids=list(range(args.num_trajectories)),
            trajectory_batch_size=args.trajectory_batch_size,
            snapshot_every=args.snapshot_every,
            verifier=verifier,
            denoise_config=config,
        )
        trajectories.extend(new_records)
        completed_ordinals.add(ordinal)
        newly_completed += 1
        if newly_completed % args.save_every == 0:
            elapsed = prior_elapsed + time.time() - started
            atomic_torch_save(
                make_shard_payload(
                    args=args,
                    shard_ordinals=shard_ordinals,
                    manifest=manifest,
                    trajectories=trajectories,
                    elapsed=elapsed,
                    complete=False,
                ),
                destination,
            )
            total = len(trajectories)
            correct = sum(bool(record["is_correct"]) for record in trajectories)
            LOGGER.info(
                "[%d/%d] ordinal=%d trajectories=%d correct=%d (%.2f%%) "
                "physical_forwards=%d generation_seconds=%.1f",
                len(completed_ordinals),
                len(shard_ordinals),
                ordinal,
                total,
                correct,
                100 * correct / total,
                execution["physical_denoising_forwards"],
                generation_seconds,
            )

    elapsed = prior_elapsed + time.time() - started
    payload = make_shard_payload(
        args=args,
        shard_ordinals=shard_ordinals,
        manifest=manifest,
        trajectories=trajectories,
        elapsed=elapsed,
        complete=completed_ordinals == set(shard_ordinals),
    )
    atomic_torch_save(payload, destination)
    total = len(trajectories)
    correct = sum(bool(record["is_correct"]) for record in trajectories)
    LOGGER.info(
        "DONE shard=%d/%d problems=%d trajectories=%d correct=%d (%.2f%%) path=%s",
        args.shard_id,
        args.num_shards,
        len(completed_ordinals),
        total,
        correct,
        100 * correct / total,
        destination,
    )


if __name__ == "__main__":
    main()
