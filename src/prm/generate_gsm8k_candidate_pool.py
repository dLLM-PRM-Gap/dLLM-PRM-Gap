"""Generate and merge the canonical unguided Dream GSM8K candidate pool.

Formal pools are restricted to N=8 with base seeds 42/43/44, plus the N=32
seed-42 prefix extension.  Generation uses the raw lm-eval-style 8-shot prompt
and the frozen Dream maskgit_plus protocol (T=128, temperature=0.5,
alg_temp=0.5, top_p=1, gen_length=512).

Formal shard (repeat ``shard_id`` over all workers)::

    python -m src.prm.generate_gsm8k_candidate_pool \
      --split test --seed 42 --num_candidates 8 \
      --model_path MODEL --model_sha256 SHA --output_dir OUT \
      --shard_id 0 --num_shards 72 --formal_run

Formal N=32 uses the merged seed-42 N=8 JSON via ``--prefix_pool`` and its
file digest via ``--prefix_pool_sha256``.  Merge with the identical pool
arguments plus ``--merge_only --expected_problems 1319 --formal_run``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


RELEASE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = RELEASE_ROOT.parent
if __package__ in {None, ""}:
    sys.path.insert(0, str(RELEASE_ROOT / "src"))
for candidate in (
    PROJECT_ROOT / "external" / "CoDD" / "lm-evaluation-harness",
    PROJECT_ROOT / "external" / "CoDD" / "eval",
    PROJECT_ROOT / "external" / "CoDD",
):
    if candidate.exists():
        sys.path.insert(0, str(candidate))

if __package__ in {None, ""}:
    from prm.eval_prm_guided_sharded import (
        build_gsm8k_prompt,
        extract_gsm8k_answer,
        normalize_answer,
    )
    from prm.gsm8k_candidate_pool import (
        POOL_PROTOCOL_ID,
        POOL_SCHEMA_VERSION,
        build_problem_record,
        fixed_denoise_config,
        generate_candidate_records,
        majority_readout,
        sampler_manifest_config,
        summarize_pool,
        validate_candidate_records,
        validate_pool_spec,
    )
    from prm.protocol import (
        artifact_descriptor,
        atomic_write_json,
        build_protocol_manifest,
        sha256_text,
        validate_artifact_attestation,
    )
else:
    from .eval_prm_guided_sharded import (
        build_gsm8k_prompt,
        extract_gsm8k_answer,
        normalize_answer,
    )
    from .gsm8k_candidate_pool import (
        POOL_PROTOCOL_ID,
        POOL_SCHEMA_VERSION,
        build_problem_record,
        fixed_denoise_config,
        generate_candidate_records,
        majority_readout,
        sampler_manifest_config,
        summarize_pool,
        validate_candidate_records,
        validate_pool_spec,
    )
    from .protocol import (
        artifact_descriptor,
        atomic_write_json,
        build_protocol_manifest,
        sha256_text,
        validate_artifact_attestation,
    )


LOGGER = logging.getLogger("generate_gsm8k_candidate_pool")
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
        selected = [int(item.strip()) for item in text.split(",") if item.strip()]
        if len(selected) != len(set(selected)):
            raise ValueError("problem_ids contains duplicates")
        if any(item < 0 or item >= total for item in selected):
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


def shard_filename(args: argparse.Namespace, shard_id: int) -> str:
    return (
        f"unguided_gsm8k_{args.split}_N{args.num_candidates}_s{args.seed}"
        f"_shard{shard_id:03d}-of-{args.num_shards:03d}.json"
    )


def merged_filename(args: argparse.Namespace) -> str:
    return (
        f"unguided_gsm8k_{args.split}_N{args.num_candidates}"
        f"_s{args.seed}_merged.json"
    )


def _validate_pool_identity(
    payload: Mapping[str, Any],
    *,
    split: str,
    seed: int,
    num_candidates: int,
) -> None:
    if int(payload.get("schema_version", 0)) != POOL_SCHEMA_VERSION:
        raise ValueError("unsupported candidate-pool schema")
    expected = {
        "task": "gsm8k",
        "split": split,
        "method": "unguided_dream_candidate_pool",
        "protocol_id": POOL_PROTOCOL_ID,
        "seed": int(seed),
        "num_candidates": int(num_candidates),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"candidate-pool {key} mismatch: {payload.get(key)!r} != {value!r}"
            )


def load_prefix_pool(
    path: str | os.PathLike[str],
    *,
    split: str,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    _validate_pool_identity(
        payload,
        split=split,
        seed=42,
        num_candidates=8,
    )
    if payload.get("complete") is not True:
        raise ValueError("prefix pool is incomplete")
    records = payload.get("per_problem")
    if not isinstance(records, list) or not records:
        raise ValueError("prefix pool has no per_problem records")
    by_id: dict[int, dict[str, Any]] = {}
    for raw in records:
        record = dict(raw)
        problem_id = int(record["global_problem_id"])
        if problem_id in by_id:
            raise ValueError(f"duplicate problem {problem_id} in prefix pool")
        validate_candidate_records(
            record["candidates"],
            base_seed=42,
            global_problem_id=problem_id,
            expected_count=8,
            gold_answer=record.get("gold_answer"),
            gen_length=512,
        )
        by_id[problem_id] = record
    return by_id, payload


def _validate_existing_shard(
    payload: Mapping[str, Any],
    args: argparse.Namespace,
    shard_problem_ids: Sequence[int],
) -> None:
    _validate_pool_identity(
        payload,
        split=args.split,
        seed=args.seed,
        num_candidates=args.num_candidates,
    )
    if int(payload.get("shard_id", -1)) != args.shard_id:
        raise ValueError("resume shard_id mismatch")
    if int(payload.get("num_shards", -1)) != args.num_shards:
        raise ValueError("resume num_shards mismatch")
    if [int(item) for item in payload.get("selected_problem_ids", [])] != list(
        shard_problem_ids
    ):
        raise ValueError("resume selected_problem_ids mismatch")
    records = payload.get("per_problem")
    if not isinstance(records, list):
        raise ValueError("resume shard per_problem must be a list")
    completed_ids = [int(item["global_problem_id"]) for item in records]
    if len(completed_ids) != len(set(completed_ids)):
        raise ValueError("resume shard contains duplicate problems")
    if not set(completed_ids).issubset(set(shard_problem_ids)):
        raise ValueError("resume shard contains a problem outside its assignment")
    for record in records:
        problem_id = int(record["global_problem_id"])
        if int(record.get("base_seed", -1)) != args.seed:
            raise ValueError(f"resume problem {problem_id} base_seed mismatch")
        if int(record.get("num_candidates", -1)) != args.num_candidates:
            raise ValueError(
                f"resume problem {problem_id} num_candidates mismatch"
            )
        prompt_ids = record.get("prompt_token_ids")
        if not isinstance(prompt_ids, list) or not prompt_ids:
            raise ValueError(f"resume problem {problem_id} has no prompt ids")
        if int(record.get("prompt_length", -1)) != len(prompt_ids):
            raise ValueError(
                f"resume problem {problem_id} prompt_length mismatch"
            )
        validate_candidate_records(
            record["candidates"],
            base_seed=args.seed,
            global_problem_id=problem_id,
            expected_count=args.num_candidates,
            gold_answer=record.get("gold_answer"),
            gen_length=512,
        )


def make_shard_payload(
    *,
    args: argparse.Namespace,
    shard_problem_ids: Sequence[int],
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    elapsed: float,
    complete: bool,
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda item: int(item["global_problem_id"]))
    return {
        "schema_version": POOL_SCHEMA_VERSION,
        "task": "gsm8k",
        "split": args.split,
        "method": "unguided_dream_candidate_pool",
        "protocol_id": POOL_PROTOCOL_ID,
        "seed": int(args.seed),
        "num_candidates": int(args.num_candidates),
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "selected_problem_ids": [int(item) for item in shard_problem_ids],
        "complete": bool(complete),
        "manifest": dict(manifest),
        "summary": summarize_pool(ordered),
        "per_problem": ordered,
        "wall_time_seconds": float(elapsed),
    }


def merge_shard_payloads(
    shards: Sequence[Mapping[str, Any]],
    *,
    split: str,
    seed: int,
    num_candidates: int,
    expected_problems: int | None,
    formal_run: bool,
) -> dict[str, Any]:
    if not shards:
        raise ValueError("no shards to merge")
    records: list[dict[str, Any]] = []
    manifests = []
    expected_num_shards = len(shards)
    for expected_shard_id, shard in enumerate(shards):
        _validate_pool_identity(
            shard,
            split=split,
            seed=seed,
            num_candidates=num_candidates,
        )
        if int(shard.get("shard_id", -1)) != expected_shard_id:
            raise ValueError("shard ids are not contiguous/in order")
        if int(shard.get("num_shards", -1)) != expected_num_shards:
            raise ValueError("shard num_shards mismatch")
        if not shard.get("complete"):
            raise ValueError(f"shard {expected_shard_id} is incomplete")
        records.extend(dict(item) for item in shard["per_problem"])
        manifests.append(dict(shard["manifest"]))

    ids = [int(item["global_problem_id"]) for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate global_problem_id across candidate-pool shards")
    records.sort(key=lambda item: int(item["global_problem_id"]))
    sorted_ids = [int(item["global_problem_id"]) for item in records]
    if expected_problems is not None and len(records) != expected_problems:
        raise ValueError(
            f"merged {len(records)} problems, expected {expected_problems}"
        )
    if formal_run and sorted_ids != list(range(1319)):
        raise ValueError("formal GSM8K pool must cover exactly problem ids 0..1318")
    for record in records:
        problem_id = int(record["global_problem_id"])
        if int(record.get("base_seed", -1)) != seed:
            raise ValueError(f"problem {problem_id} base_seed mismatch")
        if int(record.get("num_candidates", -1)) != num_candidates:
            raise ValueError(f"problem {problem_id} num_candidates mismatch")
        prompt_ids = record.get("prompt_token_ids")
        if not isinstance(prompt_ids, list) or not prompt_ids:
            raise ValueError(f"problem {problem_id} has no prompt_token_ids")
        if int(record.get("prompt_length", -1)) != len(prompt_ids):
            raise ValueError(f"problem {problem_id} prompt_length mismatch")
        validate_candidate_records(
            record["candidates"],
            base_seed=seed,
            global_problem_id=problem_id,
            expected_count=num_candidates,
            gold_answer=record.get("gold_answer"),
            gen_length=512,
        )
        expected_majority = majority_readout(
            record["candidates"], record.get("gold_answer")
        )
        for key in (
            "selected_answer",
            "selected_sample_id",
            "correct",
            "fallback_used",
            "valid_candidate_count",
            "invalid_candidate_count",
            "unique_extracted_answers",
        ):
            if record.get("majority_readout", {}).get(key) != (
                expected_majority[key]
            ):
                raise ValueError(
                    f"problem {problem_id} majority {key} mismatch"
                )
        expected_oracle = any(
            bool(candidate["correct"]) for candidate in record["candidates"]
        )
        if bool(record.get("oracle_correct")) != expected_oracle:
            raise ValueError(f"problem {problem_id} oracle metadata mismatch")

    reference = manifests[0]
    reference_model = reference.get("artifacts", {}).get("dream_backbone")
    reference_prefix = reference.get("artifacts", {}).get("n8_prefix_pool")
    reference_sampler = reference.get("config", {}).get("sampler")
    for manifest in manifests[1:]:
        if manifest.get("artifacts", {}).get("dream_backbone") != reference_model:
            raise ValueError("Dream artifact differs across shards")
        if manifest.get("artifacts", {}).get("n8_prefix_pool") != reference_prefix:
            raise ValueError("N=8 prefix artifact differs across shards")
        if manifest.get("config", {}).get("sampler") != reference_sampler:
            raise ValueError("sampler config differs across shards")
        if manifest.get("git", {}).get("commit") != reference.get("git", {}).get(
            "commit"
        ):
            raise ValueError("source commit differs across shards")
        if manifest.get("prompt_template_sha256") != reference.get(
            "prompt_template_sha256"
        ):
            raise ValueError("prompt-template code differs across shards")
        if manifest.get("extraction_callable_sha256") != reference.get(
            "extraction_callable_sha256"
        ):
            raise ValueError("answer extraction code differs across shards")
        if manifest.get("extra", {}).get("dataset_fingerprint") != (
            reference.get("extra", {}).get("dataset_fingerprint")
        ):
            raise ValueError("dataset fingerprint differs across shards")
    if formal_run:
        for manifest in manifests:
            git = manifest.get("git", {})
            if not git.get("commit") or git.get("dirty") is not False:
                raise ValueError(
                    "formal merge requires clean committed generation manifests"
                )
            dream_artifact = manifest.get("artifacts", {}).get(
                "dream_backbone", {}
            )
            if not dream_artifact.get("sha256"):
                raise ValueError(
                    "formal merge requires a hashed Dream backbone artifact"
                )
            if (
                manifest.get("extra", {}).get("prompt_mode")
                != "raw_8shot_no_chat_template"
            ):
                raise ValueError("formal merge requires the raw 8-shot prompt")
            if (seed, num_candidates) == (42, 32):
                prefix_artifact = manifest.get("artifacts", {}).get(
                    "n8_prefix_pool", {}
                )
                if not prefix_artifact.get("sha256"):
                    raise ValueError(
                        "formal N=32 merge requires a hashed N=8 prefix pool"
                    )

    return {
        "schema_version": POOL_SCHEMA_VERSION,
        "task": "gsm8k",
        "split": split,
        "method": "unguided_dream_candidate_pool",
        "protocol_id": POOL_PROTOCOL_ID,
        "seed": int(seed),
        "num_candidates": int(num_candidates),
        "num_shards": expected_num_shards,
        "complete": True,
        "summary": summarize_pool(records),
        "per_problem": records,
        "shard_manifests": manifests,
    }


def merge_shards(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    shards = []
    for shard_id in range(args.num_shards):
        path = output_dir / shard_filename(args, shard_id)
        if not path.is_file():
            raise FileNotFoundError(f"missing candidate-pool shard: {path}")
        with open(path, encoding="utf-8") as handle:
            shards.append(json.load(handle))
    merged = merge_shard_payloads(
        shards,
        split=args.split,
        seed=args.seed,
        num_candidates=args.num_candidates,
        expected_problems=args.expected_problems,
        formal_run=args.formal_run,
    )
    destination = output_dir / merged_filename(args)
    atomic_write_json(destination, merged)
    LOGGER.info(
        "MERGED %d problems N=%d seed=%d majority=%.2f%% oracle=%.2f%%",
        merged["summary"]["num_problems"],
        args.num_candidates,
        args.seed,
        100 * merged["summary"]["majority_accuracy"],
        100 * merged["summary"]["oracle_accuracy"],
    )
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--model_sha256", default=None)
    parser.add_argument("--model_hash_attestation", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--num_candidates", type=int, choices=(8, 32), default=8)
    parser.add_argument("--candidate_batch_size", type=int, default=8)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--problem_ids", default=None)
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prefix_pool", default=None)
    parser.add_argument("--prefix_pool_sha256", default=None)
    parser.add_argument("--merge_only", action="store_true")
    parser.add_argument("--expected_problems", type=int, default=None)
    parser.add_argument("--formal_run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    validate_pool_spec(args.seed, args.num_candidates)
    if args.num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("shard_id must lie in [0, num_shards)")
    if args.candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    if args.save_every <= 0:
        raise ValueError("save_every must be positive")
    if not args.merge_only and not args.model_path:
        raise ValueError("generation requires --model_path")
    if args.prefix_pool and (args.seed, args.num_candidates) != (42, 32):
        raise ValueError("--prefix_pool is only valid for the seed-42 N=32 extension")
    if args.formal_run:
        if args.split != "test":
            raise ValueError("formal candidate pools must use GSM8K test")
        if args.problem_ids is not None or args.max_problems is not None:
            raise ValueError("formal candidate pools must cover the full test set")
        if (
            not args.merge_only
            and (not args.model_sha256 or not args.model_hash_attestation)
        ):
            raise ValueError(
                "formal generation requires --model_sha256 and "
                "--model_hash_attestation"
            )
        if (
            not args.merge_only
            and args.num_candidates == 32
            and not args.prefix_pool
        ):
            raise ValueError("formal N=32 generation must reuse --prefix_pool N=8")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge_shards(args)
        return

    examples, dataset_fingerprint = load_gsm8k(args.split)
    selected_ids = parse_problem_ids(
        args.problem_ids,
        total=len(examples),
        max_problems=args.max_problems,
    )
    if args.formal_run and len(selected_ids) != 1319:
        raise ValueError("formal selection must contain all 1319 GSM8K test problems")
    shard_problem_ids = selected_ids[args.shard_id :: args.num_shards]
    if not shard_problem_ids:
        raise ValueError("this shard received zero selected problems")

    prefix_by_id: dict[int, dict[str, Any]] = {}
    prefix_payload: dict[str, Any] | None = None
    if args.prefix_pool:
        prefix_by_id, prefix_payload = load_prefix_pool(
            args.prefix_pool,
            split=args.split,
        )
        if args.formal_run and sorted(prefix_by_id) != list(range(1319)):
            raise ValueError(
                "formal N=32 generation requires a complete 1319-problem N=8 pool"
            )
        missing = sorted(set(shard_problem_ids) - set(prefix_by_id))
        if missing:
            raise ValueError(
                f"prefix pool is missing {len(missing)} shard problems; first={missing[:5]}"
            )

    denoise_config = fixed_denoise_config()
    model_artifact = artifact_descriptor(
        args.model_path,
        sha256=args.model_sha256,
        require_hash=args.formal_run,
        verify_supplied_hash=False,
    )
    artifacts: dict[str, Mapping[str, Any]] = {
        "dream_backbone": model_artifact
    }
    if args.formal_run:
        model_attestation = validate_artifact_attestation(
            args.model_hash_attestation,
            artifact_path=args.model_path,
            expected_sha256=args.model_sha256,
        )
        artifacts["dream_backbone_hash_attestation"] = model_attestation[
            "attestation"
        ]
    if args.prefix_pool:
        artifacts["n8_prefix_pool"] = artifact_descriptor(
            args.prefix_pool,
            sha256=args.prefix_pool_sha256,
            require_hash=args.formal_run,
            verify_supplied_hash=args.prefix_pool_sha256 is not None,
        )
        assert prefix_payload is not None
        prefix_manifests = prefix_payload.get("shard_manifests")
        if prefix_manifests is None and isinstance(
            prefix_payload.get("manifest"), Mapping
        ):
            prefix_manifests = [prefix_payload["manifest"]]
        if not isinstance(prefix_manifests, list) or not prefix_manifests:
            raise ValueError("N=8 prefix pool is missing generation manifests")
        expected_sampler = sampler_manifest_config(denoise_config)
        for prefix_manifest in prefix_manifests:
            if prefix_manifest.get("config", {}).get("sampler") != (
                expected_sampler
            ):
                raise ValueError("N=8 prefix pool sampler is non-canonical")
            if (
                prefix_manifest.get("extra", {}).get("prompt_mode")
                != "raw_8shot_no_chat_template"
            ):
                raise ValueError("N=8 prefix pool prompt mode is non-canonical")
            prefix_model = prefix_manifest.get("artifacts", {}).get(
                "dream_backbone", {}
            )
            if (
                model_artifact.get("sha256")
                and prefix_model.get("sha256")
                and prefix_model["sha256"] != model_artifact["sha256"]
            ):
                raise ValueError(
                    "N=8 prefix pool used a different Dream backbone"
                )
            if args.formal_run:
                git = prefix_manifest.get("git", {})
                if not git.get("commit") or git.get("dirty") is not False:
                    raise ValueError(
                        "formal N=32 generation requires clean prefix manifests"
                    )
                if not prefix_model.get("sha256"):
                    raise ValueError(
                        "formal N=32 generation requires a hashed prefix model"
                    )
                if prefix_model["sha256"] != model_artifact["sha256"]:
                    raise ValueError(
                        "formal N=32 generation model differs from N=8 prefix"
                    )
    selected_question_hash = sha256_text(
        json.dumps(
            [examples[index]["question"] for index in selected_ids],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    manifest = build_protocol_manifest(
        repo_root=RELEASE_ROOT,
        experiment="gsm8k_unguided_candidate_pool",
        config={
            "protocol_id": POOL_PROTOCOL_ID,
            "split": args.split,
            "seed": args.seed,
            "num_candidates": args.num_candidates,
            "candidate_batch_size": args.candidate_batch_size,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "shard_problem_ids": shard_problem_ids,
            "sampler": sampler_manifest_config(denoise_config),
            "prefix_candidates": 8 if args.prefix_pool else 0,
        },
        artifacts=artifacts,
        prompt_template=build_gsm8k_prompt("{QUESTION}"),
        extraction_fn=extract_gsm8k_answer,
        extra={
            "dataset_name": "gsm8k",
            "dataset_subset": "main",
            "dataset_split": args.split,
            "dataset_fingerprint": dataset_fingerprint,
            "selected_question_list_sha256": selected_question_hash,
            "prompt_mode": "raw_8shot_no_chat_template",
            "candidate_seed_key": (
                "(base_seed, global_problem_id, sample_id, "
                "'unguided_gsm8k_candidate_v1')"
            ),
            "n32_prefix_contract": (
                "seed42/N32 sample_ids 0..7 are copied from canonical seed42/N8"
            ),
        },
    )
    if args.formal_run:
        git = manifest.get("git", {})
        if not git.get("commit") or git.get("dirty") is not False:
            raise ValueError(
                "formal generation requires a clean committed release tree"
            )

    destination = output_dir / shard_filename(args, args.shard_id)
    records: list[dict[str, Any]] = []
    completed: set[int] = set()
    prior_elapsed = 0.0
    if destination.exists():
        if not args.resume:
            raise FileExistsError(
                f"{destination} exists; pass --resume or choose a new output directory"
            )
        with open(destination, encoding="utf-8") as handle:
            existing = json.load(handle)
        _validate_existing_shard(existing, args, shard_problem_ids)
        existing_commit = existing.get("manifest", {}).get("git", {}).get("commit")
        current_commit = manifest.get("git", {}).get("commit")
        if existing_commit != current_commit:
            raise ValueError(
                f"resume commit mismatch: {existing_commit} != {current_commit}"
            )
        existing_manifest = existing.get("manifest", {})
        for artifact_name in ("dream_backbone", "n8_prefix_pool"):
            if existing_manifest.get("artifacts", {}).get(artifact_name) != (
                manifest.get("artifacts", {}).get(artifact_name)
            ):
                raise ValueError(
                    f"resume artifact mismatch for {artifact_name}"
                )
        if existing_manifest.get("config", {}).get("sampler") != (
            manifest.get("config", {}).get("sampler")
        ):
            raise ValueError("resume sampler manifest mismatch")
        if existing_manifest.get("prompt_template_sha256") != (
            manifest.get("prompt_template_sha256")
        ):
            raise ValueError("resume prompt-template hash mismatch")
        if existing_manifest.get("extraction_callable_sha256") != (
            manifest.get("extraction_callable_sha256")
        ):
            raise ValueError("resume extraction-code hash mismatch")
        manifest = existing["manifest"]
        records = [dict(item) for item in existing.get("per_problem", [])]
        completed = {int(item["global_problem_id"]) for item in records}
        prior_elapsed = float(existing.get("wall_time_seconds", 0.0))
        LOGGER.info("Resuming with %d completed problems", len(completed))

    if completed == set(shard_problem_ids):
        atomic_write_json(
            destination,
            make_shard_payload(
                args=args,
                shard_problem_ids=shard_problem_ids,
                manifest=manifest,
                records=records,
                elapsed=prior_elapsed,
                complete=True,
            ),
        )
        LOGGER.info("Candidate-pool shard is already complete: %s", destination)
        return

    # All formal source/artifact and resume gates run before allocating 7B.
    from dream.modeling_dream import DreamModel
    from transformers import AutoTokenizer

    LOGGER.info(
        "Loading Dream from %s for N=%d seed=%d shard=%d/%d (%d problems)",
        args.model_path,
        args.num_candidates,
        args.seed,
        args.shard_id,
        args.num_shards,
        len(shard_problem_ids),
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

    started = time.time()
    for ordinal, global_id in enumerate(shard_problem_ids, start=1):
        if global_id in completed:
            continue
        example = examples[global_id]
        prompt_text = build_gsm8k_prompt(example["question"])
        prompt_input_ids = tokenizer.encode(
            prompt_text, return_tensors="pt"
        ).to("cuda")
        prompt_token_ids = prompt_input_ids[0].detach().cpu().tolist()

        prefix_candidates: list[dict[str, Any]] = []
        if prefix_by_id:
            prefix_record = prefix_by_id[global_id]
            if prefix_record.get("question") != example["question"]:
                raise ValueError(f"prefix question mismatch for problem {global_id}")
            if prefix_record.get("gold_answer") != example["gold_answer"]:
                raise ValueError(f"prefix gold mismatch for problem {global_id}")
            if prefix_record.get("prompt_sha256") != sha256_text(prompt_text):
                raise ValueError(f"prefix prompt mismatch for problem {global_id}")
            if [int(value) for value in prefix_record["prompt_token_ids"]] != [
                int(value) for value in prompt_token_ids
            ]:
                raise ValueError(
                    f"prefix prompt tokenization mismatch for problem {global_id}"
                )
            prefix_candidates = [
                dict(item) for item in prefix_record["candidates"]
            ]

        existing_sample_ids = {
            int(item["sample_id"]) for item in prefix_candidates
        }
        missing_sample_ids = [
            sample_id
            for sample_id in range(args.num_candidates)
            if sample_id not in existing_sample_ids
        ]
        new_candidates, execution, generation_seconds = (
            generate_candidate_records(
                model=model,
                tokenizer=tokenizer,
                prompt_input_ids=prompt_input_ids,
                global_problem_id=global_id,
                gold_answer=example["gold_answer"],
                base_seed=args.seed,
                sample_ids=missing_sample_ids,
                batch_size=args.candidate_batch_size,
                extraction_fn=extract_gsm8k_answer,
                normalize_fn=normalize_answer,
                denoise_config=denoise_config,
            )
        )
        execution.update({
            "reused_prefix_candidates": len(prefix_candidates),
            "prefix_pool": str(Path(args.prefix_pool).resolve())
            if args.prefix_pool
            else None,
        })
        record = build_problem_record(
            global_problem_id=global_id,
            question=example["question"],
            gold_answer=example["gold_answer"],
            full_gold_solution=example["full_gold_solution"],
            prompt_text=prompt_text,
            prompt_token_ids=prompt_token_ids,
            base_seed=args.seed,
            num_candidates=args.num_candidates,
            candidates=[*prefix_candidates, *new_candidates],
            denoise_config=denoise_config,
            execution=execution,
            wall_time_seconds=generation_seconds,
        )
        records.append(record)
        completed.add(global_id)
        summary = summarize_pool(records)
        LOGGER.info(
            "[%d/%d] pid=%d majority=%.1f%% oracle=%.1f%% "
            "new=%d reused=%d time=%.1fs",
            ordinal,
            len(shard_problem_ids),
            global_id,
            100 * summary["majority_accuracy"],
            100 * summary["oracle_accuracy"],
            execution["new_candidates"],
            execution["reused_prefix_candidates"],
            generation_seconds,
        )
        if len(records) % args.save_every == 0:
            atomic_write_json(
                destination,
                make_shard_payload(
                    args=args,
                    shard_problem_ids=shard_problem_ids,
                    manifest=manifest,
                    records=records,
                    elapsed=prior_elapsed + time.time() - started,
                    complete=False,
                ),
            )

    if completed != set(shard_problem_ids):
        raise AssertionError(
            f"completed {len(completed)} of {len(shard_problem_ids)} shard problems"
        )
    atomic_write_json(
        destination,
        make_shard_payload(
            args=args,
            shard_problem_ids=shard_problem_ids,
            manifest=manifest,
            records=records,
            elapsed=prior_elapsed + time.time() - started,
            complete=True,
        ),
    )
    LOGGER.info("Saved complete candidate-pool shard to %s", destination)


if __name__ == "__main__":
    main()
