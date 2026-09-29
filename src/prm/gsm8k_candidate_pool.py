"""Canonical unguided GSM8K candidate-pool primitives.

The pool is shared by Majority, Oracle, PRM-rerank, and ORM-rerank.  Candidate
``sample_id`` values are semantic identifiers: the first eight candidates of
the seed-42 N=32 pool are exactly the seed-42 N=8 pool, independent of shard
count, traversal order, or generation micro-batch size.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import asdict
from typing import Any, Callable, Mapping, Sequence

import torch

from .protocol import derive_seed, sha256_text
from .resume_denoise import (
    DenoiseConfig,
    initialize_denoise_state,
    run_denoise,
)


POOL_SCHEMA_VERSION = 1
POOL_PROTOCOL_ID = "gsm8k_raw8shot_dream_maskgitplus_t128_g512_v1"
ALLOWED_POOL_SPECS = frozenset(
    {(42, 8), (43, 8), (44, 8), (42, 32)}
)
CANDIDATE_SEED_STAGE = "unguided_gsm8k_candidate_v1"


def fixed_denoise_config() -> DenoiseConfig:
    """Return a fresh copy of the frozen submitted Dream sampling protocol."""
    return DenoiseConfig(
        steps=128,
        temperature=0.5,
        top_p=1.0,
        top_k=None,
        alg="maskgit_plus",
        alg_temp=0.5,
        mask_token_id=151666,
        gen_length=512,
        eps=1e-5,
    )


def validate_pool_spec(base_seed: int, num_candidates: int) -> None:
    spec = (int(base_seed), int(num_candidates))
    if spec not in ALLOWED_POOL_SPECS:
        raise ValueError(
            f"unsupported canonical pool spec {spec}; expected one of "
            f"{sorted(ALLOWED_POOL_SPECS)}"
        )


def candidate_seed(
    base_seed: int,
    global_problem_id: int | str,
    sample_id: int,
) -> int:
    """Stable seed for one independent unguided Dream trajectory."""
    return derive_seed(
        int(base_seed),
        global_problem_id,
        int(sample_id),
        CANDIDATE_SEED_STAGE,
    )


def majority_readout(
    candidates: Sequence[Mapping[str, Any]],
    gold_answer: str | None,
) -> dict[str, Any]:
    """Majority over valid extracted answers with deterministic fallback.

    Extraction failures are diagnostics, not an answer class: several invalid
    strings must never combine into a winning ``None`` cluster.  If every
    candidate is invalid, the lowest sample id is reported as a total fallback
    and the readout is counted incorrect.
    """
    if not candidates:
        raise ValueError("majority requires at least one candidate")
    ordered = sorted(candidates, key=lambda item: int(item["sample_id"]))
    clusters: OrderedDict[str, dict[str, Any]] = OrderedDict()
    invalid_sample_ids: list[int] = []
    for candidate in ordered:
        answer = candidate.get("extracted_answer")
        if answer is None:
            invalid_sample_ids.append(int(candidate["sample_id"]))
            continue
        key = str(answer)
        cluster = clusters.setdefault(
            key,
            {
                "key": key,
                "answer": answer,
                "sample_ids": [],
                "count": 0,
            },
        )
        cluster["sample_ids"].append(int(candidate["sample_id"]))
        cluster["count"] += 1

    valid_count = len(ordered) - len(invalid_sample_ids)
    probabilities = [
        cluster["count"] / valid_count for cluster in clusters.values()
    ] if valid_count else []
    entropy = -sum(
        probability * math.log2(probability)
        for probability in probabilities
        if probability > 0
    )
    if clusters:
        # OrderedDict insertion order is lowest sample_id. Python max keeps
        # the first item on an exact tie.
        selected = max(clusters.values(), key=lambda item: int(item["count"]))
        selected_answer = selected["answer"]
        selected_key = selected["key"]
        selected_count = int(selected["count"])
        selected_sample_id = int(selected["sample_ids"][0])
        correct = selected_answer == gold_answer
        fallback_used = False
        fallback_sample_id = None
    else:
        selected_answer = None
        selected_key = None
        selected_count = 0
        selected_sample_id = int(ordered[0]["sample_id"])
        correct = False
        fallback_used = True
        fallback_sample_id = selected_sample_id
    return {
        "selected_answer": selected_answer,
        "selected_cluster_key": selected_key,
        "selected_count": selected_count,
        "selected_sample_id": selected_sample_id,
        "correct": correct,
        "tie_break": "lowest_sample_id",
        "fallback_used": fallback_used,
        "fallback_sample_id": fallback_sample_id,
        "clusters": list(clusters.values()),
        "valid_candidate_count": valid_count,
        "invalid_candidate_count": len(invalid_sample_ids),
        "invalid_sample_ids": invalid_sample_ids,
        "invalid_rate": len(invalid_sample_ids) / len(ordered),
        "unique_extracted_answers": len(clusters),
        "answer_entropy_bits": entropy,
    }


def validate_candidate_records(
    candidates: Sequence[Mapping[str, Any]],
    *,
    base_seed: int,
    global_problem_id: int | str,
    expected_count: int,
    gold_answer: str | None,
    gen_length: int | None = None,
) -> None:
    if len(candidates) != expected_count:
        raise ValueError(
            f"problem {global_problem_id} has {len(candidates)} candidates; "
            f"expected {expected_count}"
        )
    ordered = sorted(candidates, key=lambda item: int(item["sample_id"]))
    sample_ids = [int(item["sample_id"]) for item in ordered]
    if sample_ids != list(range(expected_count)):
        raise ValueError(
            f"problem {global_problem_id} sample_ids are not 0..{expected_count - 1}"
        )
    for candidate in ordered:
        sample_id = int(candidate["sample_id"])
        expected_seed = candidate_seed(base_seed, global_problem_id, sample_id)
        if int(candidate["seed"]) != expected_seed:
            raise ValueError(
                f"problem {global_problem_id} sample {sample_id} seed mismatch"
            )
        token_ids = candidate.get("generated_token_ids")
        if not isinstance(token_ids, list):
            raise TypeError("generated_token_ids must be a JSON list")
        if gen_length is not None and len(token_ids) != gen_length:
            raise ValueError(
                f"problem {global_problem_id} sample {sample_id} has "
                f"{len(token_ids)} generated tokens; expected {gen_length}"
            )
        extracted = candidate.get("extracted_answer")
        expected_correct = extracted is not None and extracted == gold_answer
        if bool(candidate.get("correct")) != bool(expected_correct):
            raise ValueError(
                f"problem {global_problem_id} sample {sample_id} correctness mismatch"
            )


@torch.no_grad()
def generate_candidate_records(
    *,
    model,
    tokenizer,
    prompt_input_ids: torch.LongTensor,
    global_problem_id: int | str,
    gold_answer: str | None,
    base_seed: int,
    sample_ids: Sequence[int],
    batch_size: int,
    extraction_fn: Callable[[str], str | None],
    normalize_fn: Callable[[str | None], str | None],
    denoise_config: DenoiseConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], float]:
    """Generate selected candidates with independent explicit row RNG streams."""
    config = denoise_config or fixed_denoise_config()
    config.validate()
    if prompt_input_ids.ndim != 2 or prompt_input_ids.shape[0] != 1:
        raise ValueError("prompt_input_ids must have shape [1, prompt_length]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    selected = [int(value) for value in sample_ids]
    if len(selected) != len(set(selected)) or any(value < 0 for value in selected):
        raise ValueError("sample_ids must be distinct non-negative integers")

    prompt_length = int(prompt_input_ids.shape[-1])
    records: list[dict[str, Any]] = []
    physical_forwards = 0
    logical_forwards = 0
    started = time.time()
    for start in range(0, len(selected), batch_size):
        batch_ids = selected[start : start + batch_size]
        row_seeds = [
            candidate_seed(base_seed, global_problem_id, sample_id)
            for sample_id in batch_ids
        ]
        prompts = prompt_input_ids.expand(len(batch_ids), -1).clone()
        initial = initialize_denoise_state(prompts, config, row_seeds)
        completed, _ = run_denoise(model, initial, config)
        expected_logical = len(batch_ids) * config.steps
        if completed.logical_denoising_calls != expected_logical:
            raise AssertionError(
                f"logical calls {completed.logical_denoising_calls} "
                f"!= expected {expected_logical}"
            )
        if completed.physical_forward_calls != config.steps:
            raise AssertionError(
                f"physical calls {completed.physical_forward_calls} "
                f"!= expected {config.steps}"
            )
        physical_forwards += completed.physical_forward_calls
        logical_forwards += completed.logical_denoising_calls

        for row, (sample_id, seed) in enumerate(zip(batch_ids, row_seeds)):
            generated = (
                completed.x[row, prompt_length:].detach().cpu().to(torch.long)
            )
            remaining_masks = int(
                (generated == config.mask_token_id).sum().item()
            )
            if remaining_masks:
                raise AssertionError(
                    f"sample {sample_id} ended with {remaining_masks} MASK tokens"
                )
            token_ids = generated.tolist()
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            answer = normalize_fn(extraction_fn(text))
            records.append(
                {
                    "sample_id": sample_id,
                    "seed": int(seed),
                    "generated_token_ids": token_ids,
                    "generated_text": text,
                    "extracted_answer": answer,
                    "correct": answer is not None and answer == gold_answer,
                    "call_counts": {
                        "logical_denoising": config.steps,
                        "logical_scorer": 0,
                        "logical_total": config.steps,
                    },
                }
            )

    records.sort(key=lambda item: int(item["sample_id"]))
    return (
        records,
        {
            "new_candidates": len(records),
            "logical_denoising": logical_forwards,
            "physical_denoising_forwards": physical_forwards,
        },
        time.time() - started,
    )


def build_problem_record(
    *,
    global_problem_id: int,
    question: str,
    gold_answer: str | None,
    full_gold_solution: str,
    prompt_text: str,
    prompt_token_ids: Sequence[int],
    base_seed: int,
    num_candidates: int,
    candidates: Sequence[Mapping[str, Any]],
    denoise_config: DenoiseConfig,
    execution: Mapping[str, Any],
    wall_time_seconds: float,
) -> dict[str, Any]:
    normalized_candidates = [
        dict(candidate)
        for candidate in sorted(
            candidates, key=lambda item: int(item["sample_id"])
        )
    ]
    validate_candidate_records(
        normalized_candidates,
        base_seed=base_seed,
        global_problem_id=global_problem_id,
        expected_count=num_candidates,
        gold_answer=gold_answer,
        gen_length=denoise_config.gen_length,
    )
    majority = majority_readout(normalized_candidates, gold_answer)
    return {
        "global_problem_id": int(global_problem_id),
        "question": str(question),
        "gold_answer": gold_answer,
        "full_gold_solution": str(full_gold_solution),
        "prompt_sha256": sha256_text(prompt_text),
        "prompt_length": len(prompt_token_ids),
        "prompt_token_ids": [int(value) for value in prompt_token_ids],
        "base_seed": int(base_seed),
        "num_candidates": int(num_candidates),
        "majority_readout": majority,
        "oracle_correct": any(
            bool(candidate["correct"]) for candidate in normalized_candidates
        ),
        "candidates": normalized_candidates,
        "call_counts": {
            "logical_denoising": num_candidates * denoise_config.steps,
            "logical_scorer": 0,
            "logical_total": num_candidates * denoise_config.steps,
        },
        "execution": dict(execution),
        "wall_time_seconds": float(wall_time_seconds),
    }


def summarize_pool(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty candidate pool")
    count = len(records)
    candidate_counts = {int(item["num_candidates"]) for item in records}
    if len(candidate_counts) != 1:
        raise ValueError("candidate count differs across problems")
    logical_keys = ("logical_denoising", "logical_scorer", "logical_total")
    return {
        "num_problems": count,
        "num_candidates_per_problem": candidate_counts.pop(),
        "total_candidates": sum(len(item["candidates"]) for item in records),
        "majority_accuracy": sum(
            bool(item["majority_readout"]["correct"]) for item in records
        )
        / count,
        "oracle_accuracy": sum(bool(item["oracle_correct"]) for item in records)
        / count,
        "mean_unique_extracted_answers": sum(
            int(item["majority_readout"]["unique_extracted_answers"])
            for item in records
        )
        / count,
        "mean_answer_entropy_bits": sum(
            float(item["majority_readout"]["answer_entropy_bits"])
            for item in records
        )
        / count,
        "total_invalid_extractions": sum(
            int(item["majority_readout"]["invalid_candidate_count"])
            for item in records
        ),
        "mean_invalid_extraction_rate": sum(
            float(item["majority_readout"]["invalid_rate"])
            for item in records
        )
        / count,
        "logical_call_totals": {
            key: sum(int(item["call_counts"][key]) for item in records)
            for key in logical_keys
        },
        "physical_denoising_forwards_this_artifact": sum(
            int(item.get("execution", {}).get("physical_denoising_forwards", 0))
            for item in records
        ),
        "wall_time_seconds_sum": sum(
            float(item.get("wall_time_seconds", 0.0)) for item in records
        ),
    }


def sampler_manifest_config(config: DenoiseConfig | None = None) -> dict[str, Any]:
    selected = config or fixed_denoise_config()
    return asdict(selected)
