"""Analyze one canonical GSM8K pool with shared-candidate baselines.

The input is produced by :mod:`generate_gsm8k_candidate_pool`.  Majority,
Oracle, terminal DiffusionPRM reranking, and ORM reranking all consume the
same stored token sequences, so score differences cannot be attributed to
different stochastic generations.

Scoring is sharded by ``global_problem_id`` and is atomically checkpointed
after complete problems.  PRM and ORM are loaded and scored sequentially to
avoid keeping two Dream-7B backbones resident at once.

Formal shard (repeat ``shard_id`` over all workers)::

    python -m src.prm.analyze_gsm8k_candidate_pool \
      --input_pool POOL.json --output_dir OUT --methods all \
      --model_path MODEL --model_sha256 SHA \
      --prm_checkpoint PRM --prm_checkpoint_sha256 SHA \
      --prm_source_checkpoint_sha256 SHA \
      --prm_checkpoint_provenance PROVENANCE.json \
      --prm_checkpoint_equivalence_report EQUIVALENCE.json \
      --orm_checkpoint ORM --orm_checkpoint_sha256 SHA \
      --shard_id 0 --num_shards 72 --formal_run

Merge with the same pool/method/shard arguments plus
``--merge_only --expected_problems 1319 --formal_run``.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


RELEASE_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(RELEASE_ROOT / "src"))

if __package__ in {None, ""}:
    from prm.checkpointing import (
        load_diffusion_prm,
        load_state_dict_audited,
        read_prm_checkpoint,
    )
    from prm.eval_prm_guided_sharded import (
        build_gsm8k_prompt,
        extract_gsm8k_answer,
    )
    from prm.eval_prm_smc_sharded import validate_submitted_main_gate
    from prm.matched_continuation import (
        ROLES as MATCHED_SCORER_ROLES,
        validate_matched_checkpoint_gate,
    )
    from prm.gsm8k_candidate_pool import (
        POOL_PROTOCOL_ID,
        POOL_SCHEMA_VERSION,
        fixed_denoise_config,
        majority_readout,
        sampler_manifest_config,
        validate_candidate_records,
        validate_pool_spec,
    )
    from prm.protocol import (
        artifact_descriptor,
        atomic_write_json,
        build_protocol_manifest,
        validate_artifact_attestation,
    )
else:
    from .checkpointing import (
        load_diffusion_prm,
        load_state_dict_audited,
        read_prm_checkpoint,
    )
    from .eval_prm_guided_sharded import (
        build_gsm8k_prompt,
        extract_gsm8k_answer,
    )
    from .eval_prm_smc_sharded import validate_submitted_main_gate
    from .matched_continuation import (
        ROLES as MATCHED_SCORER_ROLES,
        validate_matched_checkpoint_gate,
    )
    from .gsm8k_candidate_pool import (
        POOL_PROTOCOL_ID,
        POOL_SCHEMA_VERSION,
        fixed_denoise_config,
        majority_readout,
        sampler_manifest_config,
        validate_candidate_records,
        validate_pool_spec,
    )
    from .protocol import (
        artifact_descriptor,
        atomic_write_json,
        build_protocol_manifest,
        validate_artifact_attestation,
    )


LOGGER = logging.getLogger("analyze_gsm8k_candidate_pool")
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

ANALYSIS_SCHEMA_VERSION = 1
ANALYSIS_PROTOCOL_ID = "gsm8k_shared_candidate_rerank_v1"
BASELINE_METHODS = ("majority", "oracle")
SCORER_METHODS = ("prm", "orm")
PAD_TOKEN_ID = 151643
FORMAL_EQUIVALENCE_ATOL = 1e-5
SUBMITTED_ORM_EXPECTATIONS = {
    "source_checkpoint_sha256": (
        "926e0fd890962403e9f58f6bbd663675ce8bb0b877baa88f0b9eb58f31937194"
    ),
    "adapter_sha256": (
        "217ed07d623f5e5055a311878842ec33c26adac988121a4c8a700dbff42f772c"
    ),
    "checkpoint_step": 500,
    "training_config": {
        "trajectory_dir": "data/prm_trajectories/gsm8k_combined",
        "batch_size": 8,
        "grad_accum": 4,
        "lr": 2e-5,
        "max_steps": 500,
        "seed": 42,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
    },
    "summary": {
        "train_samples": 53190,
        "val_samples": 5926,
        "best_val_acc": 0.8904826189672629,
        "total_steps": 500,
    },
}


def parse_methods(raw: str) -> tuple[str, ...]:
    """Normalize a CLI method list while always retaining both baselines."""
    requested = {value.strip().lower() for value in raw.split(",") if value.strip()}
    if "all" in requested:
        requested = set((*BASELINE_METHODS, *SCORER_METHODS))
    allowed = set((*BASELINE_METHODS, *SCORER_METHODS))
    unknown = requested - allowed
    if unknown:
        raise ValueError(f"unknown analysis methods: {sorted(unknown)}")
    requested.update(BASELINE_METHODS)
    return tuple(
        method for method in (*BASELINE_METHODS, *SCORER_METHODS) if method in requested
    )


def _core_majority_fields(readout: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: readout.get(key)
        for key in (
            "selected_answer",
            "selected_cluster_key",
            "selected_count",
            "selected_sample_id",
            "correct",
            "tie_break",
            "fallback_used",
            "fallback_sample_id",
            "valid_candidate_count",
            "invalid_candidate_count",
            "invalid_sample_ids",
            "unique_extracted_answers",
        )
    }


def validate_pool_payload(
    payload: Mapping[str, Any],
    *,
    formal_run: bool,
) -> list[dict[str, Any]]:
    """Fail closed if a pool is incomplete or departs from the fixed protocol."""
    required_identity = {
        "schema_version": POOL_SCHEMA_VERSION,
        "task": "gsm8k",
        "method": "unguided_dream_candidate_pool",
        "protocol_id": POOL_PROTOCOL_ID,
        "complete": True,
    }
    for key, expected in required_identity.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"candidate-pool {key} mismatch: {payload.get(key)!r} != {expected!r}"
            )
    split = str(payload.get("split"))
    seed = int(payload.get("seed"))
    num_candidates = int(payload.get("num_candidates"))
    validate_pool_spec(seed, num_candidates)

    raw_records = payload.get("per_problem")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("candidate pool has no per_problem records")
    records = [
        dict(record)
        for record in sorted(
            raw_records, key=lambda item: int(item["global_problem_id"])
        )
    ]
    problem_ids = [int(record["global_problem_id"]) for record in records]
    if len(problem_ids) != len(set(problem_ids)):
        raise ValueError("candidate pool contains duplicate global_problem_id")

    for record in records:
        problem_id = int(record["global_problem_id"])
        if int(record.get("base_seed", -1)) != seed:
            raise ValueError(f"problem {problem_id} base_seed mismatch")
        if int(record.get("num_candidates", -1)) != num_candidates:
            raise ValueError(f"problem {problem_id} candidate-count mismatch")
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
        recomputed = majority_readout(record["candidates"], record.get("gold_answer"))
        if _core_majority_fields(record.get("majority_readout", {})) != (
            _core_majority_fields(recomputed)
        ):
            raise ValueError(f"problem {problem_id} majority metadata mismatch")
        oracle = any(bool(item["correct"]) for item in record["candidates"])
        if bool(record.get("oracle_correct")) != oracle:
            raise ValueError(f"problem {problem_id} oracle metadata mismatch")

    manifests = payload.get("shard_manifests")
    if manifests is None and isinstance(payload.get("manifest"), Mapping):
        manifests = [payload["manifest"]]
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("candidate pool is missing its generation manifest")
    expected_sampler = sampler_manifest_config(fixed_denoise_config())
    for manifest in manifests:
        if manifest.get("config", {}).get("sampler") != expected_sampler:
            raise ValueError("candidate pool sampler manifest is non-canonical")
        prompt_mode = manifest.get("extra", {}).get("prompt_mode")
        if prompt_mode != "raw_8shot_no_chat_template":
            raise ValueError("candidate pool did not use the raw 8-shot prompt")
        if formal_run:
            git = manifest.get("git", {})
            if not git.get("commit") or git.get("dirty") is not False:
                raise ValueError(
                    "formal analysis requires clean committed pool manifests"
                )
            artifacts = manifest.get("artifacts", {})
            if not artifacts.get("dream_backbone", {}).get("sha256"):
                raise ValueError(
                    "formal analysis requires a hashed generation backbone"
                )
            if (seed, num_candidates) == (42, 32) and not artifacts.get(
                "n8_prefix_pool", {}
            ).get("sha256"):
                raise ValueError(
                    "formal N=32 analysis requires a hashed N=8 prefix pool"
                )

    if formal_run:
        if split != "test":
            raise ValueError("formal analysis requires the GSM8K test split")
        if problem_ids != list(range(1319)):
            raise ValueError(
                "formal candidate pool must cover exactly problem ids 0..1318"
            )
    return records


def select_problem_ids(
    records: Sequence[Mapping[str, Any]],
    *,
    problem_ids_text: str | None,
    max_problems: int | None,
) -> list[int]:
    available = [int(record["global_problem_id"]) for record in records]
    available_set = set(available)
    if problem_ids_text:
        selected = [
            int(value.strip()) for value in problem_ids_text.split(",") if value.strip()
        ]
        if len(selected) != len(set(selected)):
            raise ValueError("--problem_ids contains duplicates")
        missing = sorted(set(selected) - available_set)
        if missing:
            raise ValueError(
                f"requested problem ids absent from pool; first={missing[:5]}"
            )
    else:
        selected = available
    if max_problems is not None:
        if max_problems <= 0:
            raise ValueError("--max_problems must be positive")
        selected = selected[:max_problems]
    if not selected:
        raise ValueError("no problems selected for analysis")
    return selected


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _resolved_checkpoint_file(path: str | os.PathLike[str]) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_file():
        return resolved
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    for filename in ("adapter.safetensors", "model.safetensors", "best.pt"):
        candidate = resolved / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no supported checkpoint found in {resolved}")


def load_outcome_reward_model_audited(
    *,
    checkpoint: str | os.PathLike[str],
    model_path: str,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load an ORM through the same strict audited state-dict primitive."""
    from dream.modeling_dream import DreamModel

    if __package__ in {None, ""}:
        from prm.model import OutcomeRewardModel
    else:
        from .model import OutcomeRewardModel

    state_dict, config, checkpoint_format, resolved = read_prm_checkpoint(checkpoint)
    target_device = torch.device(device)
    backbone = DreamModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=dtype,
        local_files_only=True,
    )
    model = OutcomeRewardModel(
        backbone,
        hidden_size=backbone.config.hidden_size,
        lora_r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
    )
    adapter_only = checkpoint_format in {
        "safetensors_adapter",
        "pt_trainable",
    }
    missing, unexpected = load_state_dict_audited(
        model, state_dict, adapter_only=adapter_only
    )
    # ORM training/evaluation wrapped LoRA on CPU before one model-wide move;
    # reproducing that path avoids PEFT issues with an already-dispatched model.
    model.to(target_device)
    model.eval()
    report = {
        "checkpoint_path": str(resolved),
        "checkpoint_format": checkpoint_format,
        "config": dict(config),
        "supplied_keys": len(state_dict),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "model_type": "OutcomeRewardModel",
        "audited_loader": "load_state_dict_audited",
    }
    return model, report


def _load_json_mapping(
    path: str | os.PathLike[str],
    *,
    label: str,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    with open(resolved, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


def validate_submitted_orm_gate(
    *,
    checkpoint_report: Mapping[str, Any],
    adapter_sha256: str,
    source_checkpoint_sha256: str,
    provenance: Mapping[str, Any],
    equivalence: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless ORM reranking uses the audited submitted checkpoint."""
    errors: list[str] = []
    expected_source_sha256 = SUBMITTED_ORM_EXPECTATIONS["source_checkpoint_sha256"]
    expected_adapter_sha256 = SUBMITTED_ORM_EXPECTATIONS["adapter_sha256"]
    if source_checkpoint_sha256 != expected_source_sha256:
        errors.append(
            "ORM source checkpoint SHA256 is not the audited submitted checkpoint"
        )
    if adapter_sha256 != expected_adapter_sha256:
        errors.append(
            "ORM adapter SHA256 is not the audited submitted checkpoint adapter"
        )
    if checkpoint_report.get("checkpoint_format") != "safetensors_adapter":
        errors.append("formal ORM rerank requires the extracted safetensors adapter")
    if checkpoint_report.get("model_type") != "OutcomeRewardModel":
        errors.append("formal ORM load report has the wrong model_type")
    if checkpoint_report.get("audited_loader") != "load_state_dict_audited":
        errors.append("formal ORM did not use the audited checkpoint loader")
    if checkpoint_report.get("unexpected_keys"):
        errors.append("formal ORM load report contains unexpected keys")
    report_config = checkpoint_report.get("config")
    if not isinstance(report_config, Mapping):
        errors.append("formal ORM load report is missing training config")
    else:
        for key, expected in SUBMITTED_ORM_EXPECTATIONS["training_config"].items():
            actual = report_config.get(key)
            matches = (
                math.isclose(float(actual), float(expected), abs_tol=1e-12)
                if isinstance(expected, float) and isinstance(actual, (int, float))
                else actual == expected
            )
            if not matches:
                errors.append(f"ORM config {key}={actual!r}, expected {expected!r}")

    if provenance.get("source_checkpoint_sha256") != source_checkpoint_sha256:
        errors.append("ORM provenance source SHA256 does not match lock")
    if provenance.get("adapter_sha256") != adapter_sha256:
        errors.append("ORM provenance adapter SHA256 does not match evaluated file")
    if (
        provenance.get("checkpoint_step")
        != SUBMITTED_ORM_EXPECTATIONS["checkpoint_step"]
    ):
        errors.append("ORM provenance checkpoint step does not match submitted run")
    provenance_config = provenance.get("training_config")
    if not isinstance(provenance_config, Mapping):
        errors.append("ORM provenance is missing training_config")
    elif isinstance(report_config, Mapping) and dict(provenance_config) != dict(
        report_config
    ):
        errors.append("ORM provenance and adapter training configs differ")
    summary = provenance.get("source_summary")
    if not isinstance(summary, Mapping):
        errors.append("ORM provenance is missing source summary")
    else:
        for key, expected in SUBMITTED_ORM_EXPECTATIONS["summary"].items():
            actual = summary.get(key)
            matches = (
                math.isclose(float(actual), float(expected), abs_tol=1e-12)
                if isinstance(expected, float) and isinstance(actual, (int, float))
                else actual == expected
            )
            if not matches:
                errors.append(f"ORM summary {key}={actual!r}, expected {expected!r}")

    if equivalence.get("all_passed") is not True:
        errors.append("ORM full-to-adapter equivalence did not pass")
    if equivalence.get("source_checkpoint_sha256") != source_checkpoint_sha256:
        errors.append("ORM equivalence source SHA256 does not match lock")
    if equivalence.get("adapter_sha256") != adapter_sha256:
        errors.append("ORM equivalence adapter SHA256 does not match evaluated file")
    if int(equivalence.get("num_states", 0)) < 3:
        errors.append("ORM equivalence must score at least three terminal states")
    max_difference = equivalence.get("max_abs_logit_difference")
    if not isinstance(max_difference, (int, float)) or not math.isfinite(
        float(max_difference)
    ):
        errors.append("ORM equivalence max logit difference is not finite")
    elif float(max_difference) > FORMAL_EQUIVALENCE_ATOL:
        errors.append(
            f"ORM equivalence max logit difference exceeds {FORMAL_EQUIVALENCE_ATOL}"
        )
    if equivalence.get("architecture_match") is not True:
        errors.append("ORM full/adapter architecture schemas differ")
    if equivalence.get("architecture_config_match") is not True:
        errors.append("ORM full/adapter effective configs differ")
    full_audit = equivalence.get("full_load_audit")
    adapter_audit = equivalence.get("adapter_load_audit")
    if not isinstance(full_audit, Mapping) or full_audit.get("passed") is not True:
        errors.append("ORM equivalence full-checkpoint load audit did not pass")
    if (
        not isinstance(adapter_audit, Mapping)
        or adapter_audit.get("passed") is not True
        or adapter_audit.get("missing_trainable_keys") != []
    ):
        errors.append("ORM equivalence adapter load audit did not pass")

    if errors:
        raise ValueError("submitted-ORM checkpoint gate failed: " + "; ".join(errors))
    return {
        "all_passed": True,
        "expectations": SUBMITTED_ORM_EXPECTATIONS,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "adapter_sha256": adapter_sha256,
        "max_abs_logit_difference": float(max_difference),
        "formal_equivalence_atol": FORMAL_EQUIVALENCE_ATOL,
    }


def _validate_terminal_prm_report(
    report: Mapping[str, Any],
    *,
    formal_run: bool,
) -> None:
    if not formal_run:
        return
    config = report.get("config", {})
    if bool(config.get("causal", False)):
        raise ValueError("formal PRM rerank requires the bidirectional PRM")
    if bool(config.get("no_step_embed", False)):
        raise ValueError("formal PRM rerank requires step conditioning")
    if bool(config.get("no_mask_aware", False)):
        raise ValueError("formal PRM rerank requires mask-aware pooling")
    if report.get("pool_strategy") != "mean":
        raise ValueError("formal PRM rerank requires pool_strategy='mean'")


def _candidate_sequences(
    record: Mapping[str, Any],
) -> tuple[list[int], list[list[int]], list[int]]:
    prompt_ids = [int(value) for value in record["prompt_token_ids"]]
    candidates = sorted(record["candidates"], key=lambda item: int(item["sample_id"]))
    sample_ids = [int(item["sample_id"]) for item in candidates]
    sequences = [
        prompt_ids + [int(value) for value in item["generated_token_ids"]]
        for item in candidates
    ]
    return sample_ids, sequences, [len(prompt_ids)] * len(candidates)


@torch.no_grad()
def score_problem_candidates(
    *,
    model: torch.nn.Module,
    scorer: str,
    record: Mapping[str, Any],
    device: str | torch.device,
    batch_size: int,
    pad_token_id: int = PAD_TOKEN_ID,
) -> dict[str, Any]:
    """Score every stored candidate exactly once with a terminal reward model."""
    if scorer not in SCORER_METHODS:
        raise ValueError(f"unsupported scorer {scorer!r}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    target_device = torch.device(device)
    sample_ids, sequences, prompt_lengths = _candidate_sequences(record)
    scores: list[float] = []
    physical_forwards = 0
    started = time.time()
    for start in range(0, len(sequences), batch_size):
        selected = sequences[start : start + batch_size]
        selected_prompt_lengths = prompt_lengths[start : start + batch_size]
        max_length = max(len(sequence) for sequence in selected)
        input_ids = torch.full(
            (len(selected), max_length),
            int(pad_token_id),
            dtype=torch.long,
            device=target_device,
        )
        attention_mask = torch.zeros(
            (len(selected), max_length),
            dtype=torch.bool,
            device=target_device,
        )
        for row, sequence in enumerate(selected):
            length = len(sequence)
            input_ids[row, :length] = torch.tensor(
                sequence, dtype=torch.long, device=target_device
            )
            attention_mask[row, :length] = True
        prompt_lens = torch.tensor(
            selected_prompt_lengths,
            dtype=torch.long,
            device=target_device,
        )
        if scorer == "prm":
            mask_ratios = torch.zeros(
                len(selected), dtype=torch.float32, device=target_device
            )
            logits = model(
                input_ids,
                prompt_lens,
                mask_ratios,
                attention_mask=attention_mask,
            )
        else:
            logits = model(
                input_ids,
                prompt_lens,
                attention_mask=attention_mask,
            )
        batch_scores = logits.detach().float().cpu().reshape(-1).tolist()
        if len(batch_scores) != len(selected):
            raise ValueError(
                f"{scorer} returned {len(batch_scores)} scores for "
                f"{len(selected)} candidates"
            )
        if any(not math.isfinite(float(value)) for value in batch_scores):
            raise FloatingPointError(f"{scorer} produced a non-finite score")
        scores.extend(float(value) for value in batch_scores)
        physical_forwards += 1
    return {
        "sample_ids": sample_ids,
        "scores": scores,
        "call_counts": {
            "logical_scorer": len(sample_ids),
            "physical_scorer_forwards": physical_forwards,
        },
        "wall_time_seconds": time.time() - started,
    }


def validate_score_record(
    score_record: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> None:
    ordered_ids = [
        int(item["sample_id"])
        for item in sorted(candidates, key=lambda item: int(item["sample_id"]))
    ]
    if [int(value) for value in score_record.get("sample_ids", [])] != ordered_ids:
        raise ValueError("score sample_ids do not match candidate sample_ids")
    scores = score_record.get("scores")
    if not isinstance(scores, list) or len(scores) != len(ordered_ids):
        raise ValueError("score vector has the wrong length")
    if any(not math.isfinite(float(value)) for value in scores):
        raise ValueError("score vector contains non-finite values")
    calls = score_record.get("call_counts", {})
    if int(calls.get("logical_scorer", -1)) != len(ordered_ids):
        raise ValueError("logical scorer call count mismatch")
    if int(calls.get("physical_scorer_forwards", -1)) <= 0:
        raise ValueError("physical scorer call count must be positive")


def rerank_readout(
    candidates: Sequence[Mapping[str, Any]],
    score_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the maximum raw logit; exact ties use the lowest sample id."""
    validate_score_record(score_record, candidates)
    ordered = sorted(candidates, key=lambda item: int(item["sample_id"]))
    scores = [float(value) for value in score_record["scores"]]
    selected_index = max(range(len(ordered)), key=lambda index: scores[index])
    selected = ordered[selected_index]
    return {
        "selected_sample_id": int(selected["sample_id"]),
        "selected_answer": selected.get("extracted_answer"),
        "selected_raw_logit": scores[selected_index],
        "correct": bool(selected["correct"]),
        "tie_break": "lowest_sample_id",
    }


def build_analysis_problem(
    source: Mapping[str, Any],
    scorer_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a compact, auditable per-problem shared-pool comparison."""
    candidates = sorted(source["candidates"], key=lambda item: int(item["sample_id"]))
    majority = majority_readout(candidates, source.get("gold_answer"))
    oracle_ids = [
        int(candidate["sample_id"])
        for candidate in candidates
        if bool(candidate["correct"])
    ]
    candidate_rows = []
    for index, candidate in enumerate(candidates):
        row = {
            "sample_id": int(candidate["sample_id"]),
            "seed": int(candidate["seed"]),
            "extracted_answer": candidate.get("extracted_answer"),
            "correct": bool(candidate["correct"]),
            "generation_call_counts": dict(candidate.get("call_counts", {})),
            "scores": {},
        }
        for scorer, score_record in scorer_records.items():
            validate_score_record(score_record, candidates)
            row["scores"][scorer] = float(score_record["scores"][index])
        candidate_rows.append(row)

    readouts: dict[str, Any] = {
        "majority": majority,
        "oracle": {
            "correct": bool(oracle_ids),
            "correct_sample_ids": oracle_ids,
            "is_upper_bound": True,
        },
    }
    for scorer, score_record in scorer_records.items():
        readouts[scorer] = rerank_readout(candidates, score_record)

    denoising_calls = int(
        source.get("call_counts", {}).get("logical_denoising", len(candidates) * 128)
    )
    logical_by_method: dict[str, dict[str, int]] = {}
    for method in BASELINE_METHODS:
        logical_by_method[method] = {
            "logical_denoising": denoising_calls,
            "logical_scorer": 0,
            "logical_total": denoising_calls,
        }
    for scorer, score_record in scorer_records.items():
        scorer_calls = int(score_record["call_counts"]["logical_scorer"])
        logical_by_method[scorer] = {
            "logical_denoising": denoising_calls,
            "logical_scorer": scorer_calls,
            "logical_total": denoising_calls + scorer_calls,
        }

    return {
        "global_problem_id": int(source["global_problem_id"]),
        "question": source["question"],
        "gold_answer": source.get("gold_answer"),
        "prompt_sha256": source["prompt_sha256"],
        "num_candidates": len(candidates),
        "readouts": readouts,
        "candidates": candidate_rows,
        "logical_call_counts_by_method": logical_by_method,
        "physical_scorer_forwards_this_analysis": {
            scorer: int(record["call_counts"]["physical_scorer_forwards"])
            for scorer, record in scorer_records.items()
        },
        "scorer_wall_time_seconds": {
            scorer: float(record.get("wall_time_seconds", 0.0))
            for scorer, record in scorer_records.items()
        },
    }


def summarize_analysis(
    records: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty analysis")
    count = len(records)
    summary: dict[str, Any] = {
        "num_problems": count,
        "methods": list(methods),
        "accuracy": {},
        "logical_call_totals_by_method": {},
        "physical_scorer_forwards_this_analysis": {},
        "scorer_wall_time_seconds_sum": {},
        "total_invalid_extractions": sum(
            int(record["readouts"]["majority"]["invalid_candidate_count"])
            for record in records
        ),
        "mean_invalid_extraction_rate": sum(
            float(record["readouts"]["majority"]["invalid_rate"]) for record in records
        )
        / count,
    }
    for method in methods:
        summary["accuracy"][method] = (
            sum(bool(record["readouts"][method]["correct"]) for record in records)
            / count
        )
        summary["logical_call_totals_by_method"][method] = {
            key: sum(
                int(record["logical_call_counts_by_method"][method][key])
                for record in records
            )
            for key in (
                "logical_denoising",
                "logical_scorer",
                "logical_total",
            )
        }
        if method in SCORER_METHODS:
            summary["physical_scorer_forwards_this_analysis"][method] = sum(
                int(record["physical_scorer_forwards_this_analysis"].get(method, 0))
                for record in records
            )
            summary["scorer_wall_time_seconds_sum"][method] = sum(
                float(record["scorer_wall_time_seconds"].get(method, 0.0))
                for record in records
            )
    return summary


def analysis_stem(input_pool: str | os.PathLike[str], methods: Sequence[str]) -> str:
    scorers = [method for method in methods if method in SCORER_METHODS]
    suffix = "-".join(scorers) if scorers else "baselines"
    return f"{Path(input_pool).stem}_shared_rerank_{suffix}"


def shard_filename(
    input_pool: str | os.PathLike[str],
    methods: Sequence[str],
    shard_id: int,
    num_shards: int,
) -> str:
    return (
        f"{analysis_stem(input_pool, methods)}"
        f"_shard{shard_id:03d}-of-{num_shards:03d}.json"
    )


def merged_filename(input_pool: str | os.PathLike[str], methods: Sequence[str]) -> str:
    return f"{analysis_stem(input_pool, methods)}_merged.json"


def _identity_comparable(identity: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(identity)
    result.pop("shard_id", None)
    result.pop("selected_problem_ids", None)
    return result


def make_analysis_payload(
    *,
    source_pool: Mapping[str, Any],
    methods: Sequence[str],
    shard_id: int,
    num_shards: int,
    selected_problem_ids: Sequence[int],
    run_identity: Mapping[str, Any],
    manifest: Mapping[str, Any],
    scorer_state: Mapping[str, Mapping[str, Mapping[str, Any]]],
    records: Sequence[Mapping[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda item: int(item["global_problem_id"]))
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "task": "gsm8k",
        "method": "shared_candidate_pool_analysis",
        "protocol_id": ANALYSIS_PROTOCOL_ID,
        "source_pool_protocol_id": source_pool["protocol_id"],
        "split": source_pool["split"],
        "seed": int(source_pool["seed"]),
        "num_candidates": int(source_pool["num_candidates"]),
        "methods": list(methods),
        "shard_id": int(shard_id),
        "num_shards": int(num_shards),
        "selected_problem_ids": [int(value) for value in selected_problem_ids],
        "complete": bool(complete),
        "run_identity": dict(run_identity),
        "manifest": dict(manifest),
        "scorer_state": {
            method: {
                str(problem_id): dict(score_record)
                for problem_id, score_record in sorted(
                    values.items(), key=lambda item: int(item[0])
                )
            }
            for method, values in scorer_state.items()
        },
        "summary": summarize_analysis(ordered, methods) if ordered else None,
        "per_problem": ordered,
    }


def _validate_analysis_shard(
    payload: Mapping[str, Any],
    *,
    methods: Sequence[str],
    shard_id: int,
    num_shards: int,
    selected_problem_ids: Sequence[int],
    run_identity: Mapping[str, Any],
) -> None:
    expected = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "task": "gsm8k",
        "method": "shared_candidate_pool_analysis",
        "protocol_id": ANALYSIS_PROTOCOL_ID,
        "methods": list(methods),
        "shard_id": int(shard_id),
        "num_shards": int(num_shards),
        "selected_problem_ids": [int(value) for value in selected_problem_ids],
        "run_identity": dict(run_identity),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"resume analysis {key} mismatch")


def merge_analysis_payloads(
    shards: Sequence[Mapping[str, Any]],
    *,
    source_pool: Mapping[str, Any],
    methods: Sequence[str],
    expected_problem_ids: Sequence[int],
    formal_run: bool,
) -> dict[str, Any]:
    if not shards:
        raise ValueError("no analysis shards to merge")
    records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    expected_num_shards = len(shards)
    for expected_shard_id, shard in enumerate(shards):
        if int(shard.get("shard_id", -1)) != expected_shard_id:
            raise ValueError("analysis shard ids are not contiguous/in order")
        if int(shard.get("num_shards", -1)) != expected_num_shards:
            raise ValueError("analysis num_shards mismatch")
        if list(shard.get("methods", [])) != list(methods):
            raise ValueError("analysis methods differ across shards")
        if not shard.get("complete"):
            raise ValueError(f"analysis shard {expected_shard_id} is incomplete")
        records.extend(dict(record) for record in shard["per_problem"])
        manifests.append(dict(shard["manifest"]))
        identities.append(dict(shard["run_identity"]))
    reference = _identity_comparable(identities[0])
    for identity in identities[1:]:
        if _identity_comparable(identity) != reference:
            raise ValueError("analysis run identity differs across shards")
    if formal_run:
        reference_commit = manifests[0].get("git", {}).get("commit")
        for manifest in manifests:
            git = manifest.get("git", {})
            if (
                not reference_commit
                or git.get("commit") != reference_commit
                or git.get("dirty") is not False
            ):
                raise ValueError(
                    "formal merge requires one clean committed analysis source"
                )
            artifacts = manifest.get("artifacts", {})
            if not artifacts.get("candidate_pool", {}).get("sha256"):
                raise ValueError(
                    "formal merge requires a hashed candidate-pool artifact"
                )
            reports = manifest.get("extra", {}).get("checkpoint_load_reports", {})
            for scorer in (method for method in methods if method in SCORER_METHODS):
                if not artifacts.get("dream_backbone", {}).get("sha256"):
                    raise ValueError("formal scorer merge requires a hashed backbone")
                if not artifacts.get(f"{scorer}_checkpoint", {}).get("sha256"):
                    raise ValueError(
                        f"formal merge requires a hashed {scorer} checkpoint"
                    )
                if scorer not in reports:
                    raise ValueError(
                        f"formal merge is missing the audited {scorer} load report"
                    )
                if scorer == "orm":
                    if not artifacts.get("orm_checkpoint_provenance", {}).get(
                        "sha256"
                    ) or not artifacts.get("orm_checkpoint_equivalence", {}).get(
                        "sha256"
                    ):
                        raise ValueError(
                            "formal ORM merge requires hashed provenance and "
                            "equivalence artifacts"
                        )
                    orm_gate = manifest.get("extra", {}).get(
                        "submitted_orm_checkpoint_gate"
                    )
                    if (
                        not isinstance(orm_gate, Mapping)
                        or orm_gate.get("all_passed") is not True
                    ):
                        raise ValueError(
                            "formal ORM merge is missing its passing "
                            "submitted-checkpoint gate"
                        )
                if scorer == "prm":
                    scorer_role = manifest.get("extra", {}).get(
                        "prm_scorer_role", "submitted"
                    )
                    if not artifacts.get(
                        "prm_checkpoint_provenance", {}
                    ).get("sha256"):
                        raise ValueError(
                            "formal PRM merge requires hashed provenance"
                        )
                    if scorer_role == "submitted":
                        if not artifacts.get(
                            "prm_checkpoint_equivalence", {}
                        ).get("sha256"):
                            raise ValueError(
                                "formal submitted PRM merge requires equivalence"
                            )
                        prm_gate = manifest.get("extra", {}).get(
                            "submitted_prm_checkpoint_gate"
                        )
                    else:
                        if (
                            scorer_role not in MATCHED_SCORER_ROLES
                            or not artifacts.get(
                                "prm_checkpoint_pair_report", {}
                            ).get("sha256")
                        ):
                            raise ValueError(
                                "formal matched PRM merge requires a pair report"
                            )
                        prm_gate = manifest.get("extra", {}).get(
                            "matched_prm_checkpoint_gate"
                        )
                    if (
                        not isinstance(prm_gate, Mapping)
                        or prm_gate.get("all_passed") is not True
                    ):
                        raise ValueError(
                            "formal PRM merge is missing its passing checkpoint gate"
                        )

    records.sort(key=lambda item: int(item["global_problem_id"]))
    actual_ids = [int(item["global_problem_id"]) for item in records]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("duplicate problems across analysis shards")
    if actual_ids != sorted(int(value) for value in expected_problem_ids):
        raise ValueError("merged analysis does not cover the expected pool problems")
    if formal_run and actual_ids != list(range(1319)):
        raise ValueError("formal merged analysis must cover problem ids 0..1318")

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "task": "gsm8k",
        "method": "shared_candidate_pool_analysis",
        "protocol_id": ANALYSIS_PROTOCOL_ID,
        "source_pool_protocol_id": source_pool["protocol_id"],
        "split": source_pool["split"],
        "seed": int(source_pool["seed"]),
        "num_candidates": int(source_pool["num_candidates"]),
        "methods": list(methods),
        "num_shards": expected_num_shards,
        "complete": True,
        "run_identity": reference,
        "shard_manifests": manifests,
        "summary": summarize_analysis(records, methods),
        "per_problem": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input_pool", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--methods",
        default="majority,oracle",
        help="Comma list from majority,oracle,prm,orm, or 'all'.",
    )
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--prm_checkpoint", default=None)
    parser.add_argument("--orm_checkpoint", default=None)
    parser.add_argument("--pool_sha256", default=None)
    parser.add_argument("--model_sha256", default=None)
    parser.add_argument("--model_hash_attestation", default=None)
    parser.add_argument("--prm_checkpoint_sha256", default=None)
    parser.add_argument("--prm_source_checkpoint_sha256", default=None)
    parser.add_argument("--prm_checkpoint_provenance", default=None)
    parser.add_argument("--prm_checkpoint_provenance_sha256", default=None)
    parser.add_argument("--prm_checkpoint_equivalence_report", default=None)
    parser.add_argument("--prm_checkpoint_pair_report", default=None)
    parser.add_argument("--prm_checkpoint_pair_report_sha256", default=None)
    parser.add_argument(
        "--prm_scorer_role",
        choices=("submitted", *sorted(MATCHED_SCORER_ROLES)),
        default="submitted",
    )
    parser.add_argument("--orm_checkpoint_sha256", default=None)
    parser.add_argument("--orm_source_checkpoint_sha256", default=None)
    parser.add_argument("--orm_checkpoint_provenance", default=None)
    parser.add_argument("--orm_checkpoint_equivalence_report", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--problem_ids", default=None)
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge_only", action="store_true")
    parser.add_argument("--expected_problems", type=int, default=None)
    parser.add_argument("--formal_run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace, methods: Sequence[str]) -> None:
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard_id must lie in [0, num_shards)")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.save_every <= 0:
        raise ValueError("--save_every must be positive")
    scorers = [method for method in methods if method in SCORER_METHODS]
    if not args.merge_only:
        if scorers and not args.model_path:
            raise ValueError("PRM/ORM scoring requires --model_path")
        if "prm" in scorers and not args.prm_checkpoint:
            raise ValueError("PRM scoring requires --prm_checkpoint")
        if "orm" in scorers and not args.orm_checkpoint:
            raise ValueError("ORM scoring requires --orm_checkpoint")
    if args.formal_run:
        if args.problem_ids is not None or args.max_problems is not None:
            raise ValueError("formal analysis must cover the complete test pool")
        if (
            scorers
            and not args.merge_only
            and (not args.model_sha256 or not args.model_hash_attestation)
        ):
            raise ValueError(
                "formal scorer analysis requires --model_sha256 and "
                "--model_hash_attestation"
            )
        if "prm" in methods and not args.merge_only:
            if args.prm_scorer_role == "submitted":
                required_prm = (
                    args.prm_checkpoint_sha256,
                    args.prm_source_checkpoint_sha256,
                    args.prm_checkpoint_provenance,
                    args.prm_checkpoint_equivalence_report,
                )
                message = (
                    "formal submitted PRM analysis requires adapter/source "
                    "hashes, checkpoint provenance, and equivalence report"
                )
            else:
                required_prm = (
                    args.prm_checkpoint_sha256,
                    args.prm_checkpoint_provenance,
                    args.prm_checkpoint_provenance_sha256,
                    args.prm_checkpoint_pair_report,
                    args.prm_checkpoint_pair_report_sha256,
                )
                message = (
                    "formal matched PRM analysis requires checkpoint, "
                    "provenance, and pair-report hashes"
                )
            if any(not value for value in required_prm):
                raise ValueError(message)
        if (
            "orm" in methods
            and not args.merge_only
            and any(
                not value
                for value in (
                    args.orm_checkpoint_sha256,
                    args.orm_source_checkpoint_sha256,
                    args.orm_checkpoint_provenance,
                    args.orm_checkpoint_equivalence_report,
                )
            )
        ):
            raise ValueError(
                "formal ORM analysis requires adapter/source hashes, "
                "checkpoint provenance, and equivalence report"
            )


def _build_artifacts(
    args: argparse.Namespace,
    methods: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    artifacts: dict[str, Mapping[str, Any]] = {
        "candidate_pool": artifact_descriptor(
            args.input_pool,
            sha256=args.pool_sha256,
            require_hash=True,
            verify_supplied_hash=args.pool_sha256 is not None,
        )
    }
    scorers = [method for method in methods if method in SCORER_METHODS]
    if scorers:
        artifacts["dream_backbone"] = artifact_descriptor(
            args.model_path,
            sha256=args.model_sha256,
            require_hash=args.formal_run,
            verify_supplied_hash=False,
        )
        if args.formal_run:
            model_attestation = validate_artifact_attestation(
                args.model_hash_attestation,
                artifact_path=args.model_path,
                expected_sha256=args.model_sha256,
            )
            artifacts["dream_backbone_hash_attestation"] = model_attestation[
                "attestation"
            ]
    if "prm" in methods:
        path = _resolved_checkpoint_file(args.prm_checkpoint)
        artifacts["prm_checkpoint"] = artifact_descriptor(
            path,
            sha256=args.prm_checkpoint_sha256,
            require_hash=True,
            verify_supplied_hash=args.prm_checkpoint_sha256 is not None,
        )
        if args.formal_run:
            artifacts["prm_checkpoint_provenance"] = artifact_descriptor(
                args.prm_checkpoint_provenance,
                sha256=args.prm_checkpoint_provenance_sha256,
                require_hash=True,
                verify_supplied_hash=(
                    args.prm_checkpoint_provenance_sha256 is not None
                ),
            )
            if args.prm_scorer_role == "submitted":
                artifacts["prm_checkpoint_equivalence"] = artifact_descriptor(
                    args.prm_checkpoint_equivalence_report
                )
            else:
                artifacts["prm_checkpoint_pair_report"] = artifact_descriptor(
                    args.prm_checkpoint_pair_report,
                    sha256=args.prm_checkpoint_pair_report_sha256,
                    require_hash=True,
                    verify_supplied_hash=True,
                )
    if "orm" in methods:
        path = _resolved_checkpoint_file(args.orm_checkpoint)
        artifacts["orm_checkpoint"] = artifact_descriptor(
            path,
            sha256=args.orm_checkpoint_sha256,
            require_hash=True,
            verify_supplied_hash=args.orm_checkpoint_sha256 is not None,
        )
        if args.formal_run:
            artifacts["orm_checkpoint_provenance"] = artifact_descriptor(
                args.orm_checkpoint_provenance
            )
            artifacts["orm_checkpoint_equivalence"] = artifact_descriptor(
                args.orm_checkpoint_equivalence_report
            )
    return artifacts


def _run_identity(
    *,
    source_pool: Mapping[str, Any],
    methods: Sequence[str],
    artifacts: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
    selected_problem_ids: Sequence[int],
) -> dict[str, Any]:
    artifact_hashes = {
        name: artifact.get("sha256") for name, artifact in artifacts.items()
    }
    return {
        "protocol_id": ANALYSIS_PROTOCOL_ID,
        "source_pool_protocol_id": source_pool["protocol_id"],
        "source_split": source_pool["split"],
        "source_seed": int(source_pool["seed"]),
        "source_num_candidates": int(source_pool["num_candidates"]),
        "methods": list(methods),
        "artifact_sha256": artifact_hashes,
        "pad_token_id": PAD_TOKEN_ID,
        "terminal_mask_ratio": 0.0,
        "score_tie_break": "lowest_sample_id",
        "batch_size": int(args.batch_size),
        "dtype": args.dtype,
        "prm_source_checkpoint_sha256": (
            args.prm_source_checkpoint_sha256 if "prm" in methods else None
        ),
        "prm_scorer_role": (
            args.prm_scorer_role if "prm" in methods else None
        ),
        "orm_source_checkpoint_sha256": (
            args.orm_source_checkpoint_sha256 if "orm" in methods else None
        ),
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
        "selected_problem_ids": [int(value) for value in selected_problem_ids],
    }


def merge_only(
    args: argparse.Namespace,
    *,
    source_pool: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
) -> None:
    selected_ids = select_problem_ids(
        records,
        problem_ids_text=args.problem_ids,
        max_problems=args.max_problems,
    )
    if args.expected_problems is not None and (
        len(selected_ids) != args.expected_problems
    ):
        raise ValueError(
            f"pool selection has {len(selected_ids)} problems; "
            f"expected {args.expected_problems}"
        )
    output_dir = Path(args.output_dir)
    shards = []
    for shard_id in range(args.num_shards):
        path = output_dir / shard_filename(
            args.input_pool, methods, shard_id, args.num_shards
        )
        if not path.is_file():
            raise FileNotFoundError(f"missing analysis shard: {path}")
        with open(path, encoding="utf-8") as handle:
            shards.append(json.load(handle))
    merged = merge_analysis_payloads(
        shards,
        source_pool=source_pool,
        methods=methods,
        expected_problem_ids=selected_ids,
        formal_run=args.formal_run,
    )
    destination = output_dir / merged_filename(args.input_pool, methods)
    atomic_write_json(destination, merged)
    accuracies = ", ".join(
        f"{method}={100 * merged['summary']['accuracy'][method]:.2f}%"
        for method in methods
    )
    LOGGER.info("Merged %d problems: %s", len(selected_ids), accuracies)
    LOGGER.info("Saved %s", destination)


def main() -> None:
    args = build_parser().parse_args()
    methods = parse_methods(args.methods)
    validate_args(args, methods)
    with open(args.input_pool, encoding="utf-8") as handle:
        source_pool = json.load(handle)
    records = validate_pool_payload(source_pool, formal_run=args.formal_run)
    if args.merge_only:
        merge_only(
            args,
            source_pool=source_pool,
            records=records,
            methods=methods,
        )
        return

    selected_ids = select_problem_ids(
        records,
        problem_ids_text=args.problem_ids,
        max_problems=args.max_problems,
    )
    if args.formal_run and selected_ids != list(range(1319)):
        raise ValueError("formal analysis selection must be problem ids 0..1318")
    shard_ids = selected_ids[args.shard_id :: args.num_shards]
    if not shard_ids:
        raise ValueError("this analysis shard received zero problems")
    source_by_id = {int(record["global_problem_id"]): record for record in records}
    shard_records = [source_by_id[problem_id] for problem_id in shard_ids]

    artifacts = _build_artifacts(args, methods)
    run_identity = _run_identity(
        source_pool=source_pool,
        methods=methods,
        artifacts=artifacts,
        args=args,
        selected_problem_ids=shard_ids,
    )
    manifest = build_protocol_manifest(
        repo_root=RELEASE_ROOT,
        experiment="gsm8k_shared_candidate_pool_analysis",
        config={
            "protocol_id": ANALYSIS_PROTOCOL_ID,
            "methods": methods,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "device": args.device,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "shard_problem_ids": shard_ids,
            "pad_token_id": PAD_TOKEN_ID,
            "terminal_mask_ratio": 0.0,
            "score_tie_break": "lowest_sample_id",
        },
        artifacts=artifacts,
        prompt_template=build_gsm8k_prompt("{QUESTION}"),
        extraction_fn=extract_gsm8k_answer,
        extra={
            "shared_pool_contract": (
                "Majority, Oracle, PRM, and ORM consume identical stored "
                "generated_token_ids"
            ),
            "checkpoint_load_reports": {},
            "prm_scorer_role": (
                args.prm_scorer_role if "prm" in methods else None
            ),
        },
    )
    if args.formal_run:
        git = manifest.get("git", {})
        if not git.get("commit") or git.get("dirty") is not False:
            raise ValueError("formal analysis requires a clean committed release tree")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / shard_filename(
        args.input_pool, methods, args.shard_id, args.num_shards
    )
    scorer_state: dict[str, dict[int, dict[str, Any]]] = {
        method: {} for method in methods if method in SCORER_METHODS
    }
    if destination.exists():
        if not args.resume:
            raise FileExistsError(
                f"{destination} exists; pass --resume or choose another output"
            )
        with open(destination, encoding="utf-8") as handle:
            existing = json.load(handle)
        _validate_analysis_shard(
            existing,
            methods=methods,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            selected_problem_ids=shard_ids,
            run_identity=run_identity,
        )
        if existing.get("complete"):
            LOGGER.info("Analysis shard is already complete: %s", destination)
            return
        manifest = dict(existing["manifest"])
        for method, values in existing.get("scorer_state", {}).items():
            if method not in scorer_state:
                raise ValueError(f"resume has unexpected scorer state {method}")
            scorer_state[method] = {
                int(problem_id): dict(score_record)
                for problem_id, score_record in values.items()
            }
        LOGGER.info(
            "Resuming scorer counts: %s",
            {method: len(values) for method, values in scorer_state.items()},
        )

    def save_partial() -> None:
        atomic_write_json(
            destination,
            make_analysis_payload(
                source_pool=source_pool,
                methods=methods,
                shard_id=args.shard_id,
                num_shards=args.num_shards,
                selected_problem_ids=shard_ids,
                run_identity=run_identity,
                manifest=manifest,
                scorer_state=scorer_state,
                records=[],
                complete=False,
            ),
        )

    dtype = _dtype_from_name(args.dtype)
    for scorer in (method for method in SCORER_METHODS if method in methods):
        missing_records = [
            record
            for record in shard_records
            if int(record["global_problem_id"]) not in scorer_state[scorer]
        ]
        if not missing_records:
            LOGGER.info("%s scores are already complete; skipping load", scorer)
            continue
        LOGGER.info(
            "Loading audited %s checkpoint; %d problems remain",
            scorer.upper(),
            len(missing_records),
        )
        if scorer == "prm":
            model, raw_report = load_diffusion_prm(
                checkpoint=args.prm_checkpoint,
                model_path=args.model_path,
                device=args.device,
                dtype=dtype,
            )
            report = raw_report.to_dict()
            _validate_terminal_prm_report(report, formal_run=args.formal_run)
            if args.formal_run:
                prm_provenance = _load_json_mapping(
                    args.prm_checkpoint_provenance,
                    label="PRM checkpoint provenance",
                )
                if args.prm_scorer_role == "submitted":
                    prm_equivalence = _load_json_mapping(
                        args.prm_checkpoint_equivalence_report,
                        label="PRM checkpoint equivalence report",
                    )
                    gate = validate_submitted_main_gate(
                        checkpoint_report=report,
                        adapter_sha256=artifacts["prm_checkpoint"]["sha256"],
                        source_checkpoint_sha256=(args.prm_source_checkpoint_sha256),
                        provenance=prm_provenance,
                        equivalence=prm_equivalence,
                    )
                    gate_name = "submitted_prm_checkpoint_gate"
                else:
                    pair_report = _load_json_mapping(
                        args.prm_checkpoint_pair_report,
                        label="matched PRM pair report",
                    )
                    gate = validate_matched_checkpoint_gate(
                        role=args.prm_scorer_role,
                        checkpoint_sha256=artifacts["prm_checkpoint"]["sha256"],
                        checkpoint_report=report,
                        provenance=prm_provenance,
                        provenance_path=args.prm_checkpoint_provenance,
                        provenance_sha256=artifacts[
                            "prm_checkpoint_provenance"
                        ]["sha256"],
                        pair_report=pair_report,
                        pair_report_path=args.prm_checkpoint_pair_report,
                        pair_report_sha256=artifacts[
                            "prm_checkpoint_pair_report"
                        ]["sha256"],
                    )
                    gate_name = "matched_prm_checkpoint_gate"
                manifest.setdefault("extra", {})[gate_name] = gate
        else:
            model, report = load_outcome_reward_model_audited(
                checkpoint=args.orm_checkpoint,
                model_path=args.model_path,
                device=args.device,
                dtype=dtype,
            )
            if args.formal_run:
                orm_provenance = _load_json_mapping(
                    args.orm_checkpoint_provenance,
                    label="ORM checkpoint provenance",
                )
                orm_equivalence = _load_json_mapping(
                    args.orm_checkpoint_equivalence_report,
                    label="ORM checkpoint equivalence report",
                )
                manifest.setdefault("extra", {})["submitted_orm_checkpoint_gate"] = (
                    validate_submitted_orm_gate(
                        checkpoint_report=report,
                        adapter_sha256=artifacts["orm_checkpoint"]["sha256"],
                        source_checkpoint_sha256=(args.orm_source_checkpoint_sha256),
                        provenance=orm_provenance,
                        equivalence=orm_equivalence,
                    )
                )
        manifest.setdefault("extra", {}).setdefault("checkpoint_load_reports", {})[
            scorer
        ] = report
        save_partial()

        completed_since_load = 0
        for record in missing_records:
            problem_id = int(record["global_problem_id"])
            score_record = score_problem_candidates(
                model=model,
                scorer=scorer,
                record=record,
                device=args.device,
                batch_size=args.batch_size,
            )
            validate_score_record(score_record, record["candidates"])
            scorer_state[scorer][problem_id] = score_record
            completed_since_load += 1
            LOGGER.info(
                "%s pid=%d [%d/%d] time=%.2fs",
                scorer.upper(),
                problem_id,
                completed_since_load,
                len(missing_records),
                score_record["wall_time_seconds"],
            )
            if completed_since_load % args.save_every == 0:
                save_partial()
        save_partial()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    analysis_records = []
    for source in shard_records:
        problem_id = int(source["global_problem_id"])
        problem_scores = {
            scorer: scorer_state[scorer][problem_id]
            for scorer in SCORER_METHODS
            if scorer in methods
        }
        analysis_records.append(build_analysis_problem(source, problem_scores))
    final_payload = make_analysis_payload(
        source_pool=source_pool,
        methods=methods,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        selected_problem_ids=shard_ids,
        run_identity=run_identity,
        manifest=manifest,
        scorer_state=scorer_state,
        records=analysis_records,
        complete=True,
    )
    atomic_write_json(destination, final_payload)
    accuracies = ", ".join(
        f"{method}={100 * final_payload['summary']['accuracy'][method]:.2f}%"
        for method in methods
    )
    LOGGER.info("Complete analysis shard: %s", accuracies)
    LOGGER.info("Saved %s", destination)


if __name__ == "__main__":
    main()
