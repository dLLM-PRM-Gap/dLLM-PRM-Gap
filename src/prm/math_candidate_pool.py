"""Canonical shared-sampler MATH500 candidate-pool primitives.

Majority and ORM reranking must consume the exact same candidate records.
Correctness is always taken from :class:`PinnedMathVerifier` on the raw model
output; a separate strict, no-fallback ``math-verify`` parse is used for
equivalence clustering and invalid-output diagnostics.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from typing import Any, Mapping, Sequence

import torch

from .math_protocol import PinnedMathVerifier
from .protocol import derive_seed, sha256_text
from .resume_denoise import (
    DenoiseConfig,
    initialize_denoise_state,
    run_denoise,
)


POOL_SCHEMA_VERSION = 1
POOL_PROTOCOL_ID = "math500_chat_dream_maskgitplus_t128_g512_shared_v1"
CANDIDATE_SEED_STAGE = "unguided_math500_candidate_v1"
FORMAL_N8_SPECS = frozenset({(42, 8), (43, 8)})


def fixed_denoise_config() -> DenoiseConfig:
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


def validate_pool_spec(
    base_seed: int,
    num_candidates: int,
    *,
    formal_run: bool,
) -> None:
    if int(num_candidates) <= 0:
        raise ValueError("num_candidates must be positive")
    if formal_run and (int(base_seed), int(num_candidates)) not in FORMAL_N8_SPECS:
        raise ValueError(
            "formal first-stage MATH500 pools are restricted to "
            f"{sorted(FORMAL_N8_SPECS)}"
        )


def candidate_seed(
    base_seed: int,
    global_problem_id: int | str,
    sample_id: int,
) -> int:
    return derive_seed(
        int(base_seed),
        global_problem_id,
        int(sample_id),
        CANDIDATE_SEED_STAGE,
    )


def _grade_correct(candidate: Mapping[str, Any]) -> bool:
    grade = candidate.get("grade")
    return bool(grade.get("correct")) if isinstance(grade, Mapping) else False


def majority_readout(
    candidates: Sequence[Mapping[str, Any]],
    verifier: PinnedMathVerifier,
) -> dict[str, Any]:
    """Cluster valid answers by mathematical equivalence.

    Pairwise equivalence edges are closed transitively with union-find.
    Extraction failures are excluded instead of forming a ``None`` answer
    class.  Both cluster and all-invalid ties select the lowest sample id.
    """
    if not candidates:
        raise ValueError("majority requires at least one candidate")
    ordered = sorted(candidates, key=lambda item: int(item["sample_id"]))
    valid = [
        candidate
        for candidate in ordered
        if candidate.get("extracted_answer") is not None
    ]
    invalid_sample_ids = [
        int(candidate["sample_id"])
        for candidate in ordered
        if candidate.get("extracted_answer") is None
    ]

    parent = list(range(len(valid)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        # Retain the component containing the lowest ordered candidate as root.
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    comparison_errors: list[dict[str, Any]] = []
    comparison_count = 0
    for left in range(len(valid)):
        for right in range(left + 1, len(valid)):
            comparison_count += 1
            result = verifier.compare_answers(
                str(valid[left]["extracted_answer"]),
                str(valid[right]["extracted_answer"]),
            )
            if result.status != "ok":
                comparison_errors.append(
                    {
                        "left_sample_id": int(valid[left]["sample_id"]),
                        "right_sample_id": int(valid[right]["sample_id"]),
                        **result.to_dict(),
                    }
                )
            elif result.equivalent:
                union(left, right)

    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for index, candidate in enumerate(valid):
        grouped.setdefault(find(index), []).append(candidate)
    clusters: list[dict[str, Any]] = []
    for members in grouped.values():
        sorted_members = sorted(members, key=lambda item: int(item["sample_id"]))
        representative = sorted_members[0]
        clusters.append(
            {
                "representative_sample_id": int(representative["sample_id"]),
                "representative_answer": representative["extracted_answer"],
                "sample_ids": [
                    int(candidate["sample_id"]) for candidate in sorted_members
                ],
                "count": len(sorted_members),
            }
        )
    clusters.sort(key=lambda item: int(item["representative_sample_id"]))

    valid_count = len(valid)
    probabilities = (
        [int(cluster["count"]) / valid_count for cluster in clusters]
        if valid_count
        else []
    )
    entropy = -sum(
        probability * math.log2(probability)
        for probability in probabilities
        if probability > 0
    )
    if clusters:
        selected_cluster = max(
            clusters,
            key=lambda item: (
                int(item["count"]),
                -int(item["representative_sample_id"]),
            ),
        )
        selected_sample_id = int(selected_cluster["representative_sample_id"])
        selected_candidate = next(
            item for item in ordered if int(item["sample_id"]) == selected_sample_id
        )
        selected_answer = selected_cluster["representative_answer"]
        selected_count = int(selected_cluster["count"])
        correct = _grade_correct(selected_candidate)
        fallback_used = False
        fallback_sample_id = None
    else:
        selected_sample_id = int(ordered[0]["sample_id"])
        selected_answer = None
        selected_count = 0
        correct = False
        fallback_used = True
        fallback_sample_id = selected_sample_id

    return {
        "selected_answer": selected_answer,
        "selected_count": selected_count,
        "selected_sample_id": selected_sample_id,
        "correct": correct,
        "tie_break": "lowest_sample_id",
        "fallback_used": fallback_used,
        "fallback_sample_id": fallback_sample_id,
        "clusters": clusters,
        "valid_candidate_count": valid_count,
        "invalid_candidate_count": len(invalid_sample_ids),
        "invalid_sample_ids": invalid_sample_ids,
        "invalid_rate": len(invalid_sample_ids) / len(ordered),
        "unique_equivalence_classes": len(clusters),
        "answer_entropy_bits": entropy,
        "equivalence_comparison_count": comparison_count,
        "equivalence_error_count": len(comparison_errors),
        "equivalence_errors": comparison_errors,
    }


def validate_candidate_records(
    candidates: Sequence[Mapping[str, Any]],
    *,
    base_seed: int,
    global_problem_id: int | str,
    expected_count: int,
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
        grade = candidate.get("grade")
        if not isinstance(grade, Mapping):
            raise TypeError("candidate grade must be a mapping")
        if grade.get("status") not in {"ok", "timeout", "error"}:
            raise ValueError("candidate grade has invalid status")
        if bool(candidate.get("correct")) != bool(grade.get("correct")):
            raise ValueError(
                f"problem {global_problem_id} sample {sample_id} "
                "correctness differs from raw-output grade"
            )
        extraction = candidate.get("extraction")
        if not isinstance(extraction, Mapping):
            raise TypeError("candidate extraction must be a mapping")
        if extraction.get("status") not in {
            "ok",
            "invalid",
            "timeout",
            "error",
        }:
            raise ValueError("candidate extraction has invalid status")
        if candidate.get("extracted_answer") != extraction.get("answer"):
            raise ValueError("candidate extracted answer differs from parser record")


@torch.no_grad()
def generate_candidate_records(
    *,
    model,
    tokenizer,
    prompt_input_ids: torch.LongTensor,
    global_problem_id: int | str,
    gold_answer: str,
    base_seed: int,
    sample_ids: Sequence[int],
    batch_size: int,
    verifier: PinnedMathVerifier,
    denoise_config: DenoiseConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], float]:
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
        if completed.logical_denoising_calls != len(batch_ids) * config.steps:
            raise AssertionError("unexpected logical denoising call count")
        if completed.physical_forward_calls != config.steps:
            raise AssertionError("unexpected physical denoising call count")
        physical_forwards += completed.physical_forward_calls
        logical_forwards += completed.logical_denoising_calls

        for row, (sample_id, seed) in enumerate(zip(batch_ids, row_seeds)):
            generated = completed.x[row, prompt_length:].detach().cpu().to(torch.long)
            remaining_masks = int((generated == config.mask_token_id).sum().item())
            if remaining_masks:
                raise AssertionError(
                    f"sample {sample_id} ended with {remaining_masks} MASK tokens"
                )
            token_ids = generated.tolist()
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            grade = verifier.grade_output(text, gold_answer)
            extraction = verifier.extract_answer(text)
            records.append(
                {
                    "sample_id": sample_id,
                    "seed": int(seed),
                    "generated_token_ids": token_ids,
                    "generated_text": text,
                    "extracted_answer": extraction.answer,
                    "extraction": extraction.to_dict(),
                    "grade": grade.to_dict(),
                    "correct": grade.correct,
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
    global_problem_id: int | str,
    problem_ordinal: int,
    question: str,
    problem_digest: str,
    gold_answer: str,
    full_gold_solution: str,
    prompt_text: str,
    prompt_token_ids: Sequence[int],
    base_seed: int,
    num_candidates: int,
    candidates: Sequence[Mapping[str, Any]],
    denoise_config: DenoiseConfig,
    verifier: PinnedMathVerifier,
    execution: Mapping[str, Any],
    wall_time_seconds: float,
) -> dict[str, Any]:
    normalized_candidates = [
        dict(candidate)
        for candidate in sorted(candidates, key=lambda item: int(item["sample_id"]))
    ]
    validate_candidate_records(
        normalized_candidates,
        base_seed=base_seed,
        global_problem_id=global_problem_id,
        expected_count=num_candidates,
        gen_length=denoise_config.gen_length,
    )
    majority = majority_readout(normalized_candidates, verifier)
    return {
        "global_problem_id": str(global_problem_id),
        "problem_ordinal": int(problem_ordinal),
        "question": str(question),
        "problem_sha256": str(problem_digest),
        "gold_answer": str(gold_answer),
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
        "mean_unique_equivalence_classes": sum(
            int(item["majority_readout"]["unique_equivalence_classes"])
            for item in records
        )
        / count,
        "mean_answer_entropy_bits": sum(
            float(item["majority_readout"]["answer_entropy_bits"]) for item in records
        )
        / count,
        "total_invalid_extractions": sum(
            int(item["majority_readout"]["invalid_candidate_count"]) for item in records
        ),
        "total_equivalence_errors": sum(
            int(item["majority_readout"]["equivalence_error_count"]) for item in records
        ),
        "mean_invalid_extraction_rate": sum(
            float(item["majority_readout"]["invalid_rate"]) for item in records
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


def sampler_manifest_config(
    config: DenoiseConfig | None = None,
) -> dict[str, Any]:
    return asdict(config or fixed_denoise_config())
