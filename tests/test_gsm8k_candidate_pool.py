from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.prm.analyze_gsm8k_candidate_pool import (
    SUBMITTED_ORM_EXPECTATIONS,
    build_analysis_problem,
    make_analysis_payload,
    merge_analysis_payloads,
    parse_methods,
    rerank_readout,
    score_problem_candidates,
    summarize_analysis,
    validate_pool_payload,
    validate_submitted_orm_gate,
)
from src.prm.generate_gsm8k_candidate_pool import merge_shard_payloads
from src.prm.gsm8k_candidate_pool import (
    POOL_PROTOCOL_ID,
    POOL_SCHEMA_VERSION,
    build_problem_record,
    candidate_seed,
    fixed_denoise_config,
    generate_candidate_records,
    majority_readout,
    sampler_manifest_config,
)
from src.prm.resume_denoise import DenoiseConfig


class TinyDream(nn.Module):
    """Small state-dependent denoiser that exercises explicit row RNG."""

    def __init__(self, vocab_size: int, mask_token_id: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id

    def forward(self, x, attention_mask, position_ids):
        batch, length = x.shape
        vocab = torch.arange(self.vocab_size).view(1, 1, -1)
        positions = torch.arange(length).view(1, -1, 1)
        context = (x.masked_fill(x == self.mask_token_id, 0).sum(-1) % 7).view(
            batch, 1, 1
        )
        center = (positions + context) % self.vocab_size
        logits = -0.17 * (vocab - center).abs().float()
        return SimpleNamespace(logits=logits.expand(batch, -1, -1).contiguous())


class TinyTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return ",".join(str(value) for value in token_ids)


class TinyPRM(nn.Module):
    def forward(
        self,
        input_ids,
        prompt_lens,
        mask_ratios,
        attention_mask=None,
    ):
        del prompt_lens, mask_ratios
        return (input_ids * attention_mask).float().sum(-1)


class TinyORM(nn.Module):
    def forward(self, input_ids, prompt_lens, attention_mask=None):
        del prompt_lens
        return (input_ids * attention_mask).float().sum(-1)


def test_n8_is_exact_prefix_of_n32_across_microbatch_sizes():
    mask = 31
    config = DenoiseConfig(
        steps=4,
        gen_length=16,
        mask_token_id=mask,
        temperature=0.7,
        top_p=1.0,
        alg="maskgit_plus",
        alg_temp=0.5,
    )
    model = TinyDream(vocab_size=mask, mask_token_id=mask)
    tokenizer = TinyTokenizer()
    prompt = torch.tensor([[2, 3, 4]], dtype=torch.long)
    common = {
        "model": model,
        "tokenizer": tokenizer,
        "prompt_input_ids": prompt,
        "global_problem_id": 17,
        "gold_answer": "not-generated",
        "base_seed": 42,
        "extraction_fn": lambda text: text,
        "normalize_fn": lambda answer: answer,
        "denoise_config": config,
    }
    n8, calls8, _ = generate_candidate_records(
        **common, sample_ids=list(range(8)), batch_size=3
    )
    n32, calls32, _ = generate_candidate_records(
        **common, sample_ids=list(range(32)), batch_size=5
    )
    assert n32[:8] == n8
    assert [item["seed"] for item in n8] == [
        candidate_seed(42, 17, sample_id) for sample_id in range(8)
    ]
    assert calls8["logical_denoising"] == 8 * config.steps
    assert calls32["logical_denoising"] == 32 * config.steps
    assert calls8["physical_denoising_forwards"] == 3 * config.steps
    assert calls32["physical_denoising_forwards"] == 7 * config.steps


def test_majority_excludes_invalid_and_all_invalid_falls_back_incorrect():
    candidates = [
        {"sample_id": 0, "extracted_answer": None},
        {"sample_id": 1, "extracted_answer": None},
        {"sample_id": 2, "extracted_answer": "7"},
        {"sample_id": 3, "extracted_answer": "8"},
    ]
    readout = majority_readout(candidates, "7")
    assert readout["selected_answer"] == "7"
    assert readout["selected_sample_id"] == 2
    assert readout["correct"]
    assert readout["valid_candidate_count"] == 2
    assert readout["invalid_candidate_count"] == 2
    assert readout["invalid_rate"] == 0.5
    assert readout["unique_extracted_answers"] == 2
    assert readout["answer_entropy_bits"] == 1.0
    assert all(cluster["answer"] is not None for cluster in readout["clusters"])

    fallback = majority_readout(
        [
            {"sample_id": 7, "extracted_answer": None},
            {"sample_id": 2, "extracted_answer": None},
        ],
        None,
    )
    assert fallback["fallback_used"]
    assert fallback["fallback_sample_id"] == 2
    assert fallback["selected_sample_id"] == 2
    assert fallback["selected_answer"] is None
    assert not fallback["correct"]
    assert fallback["clusters"] == []
    assert fallback["unique_extracted_answers"] == 0
    assert fallback["answer_entropy_bits"] == 0


def _fake_candidates(problem_id: int, gold: str = "7"):
    candidates = []
    for sample_id in range(8):
        answer = "7" if sample_id in {0, 3} else str(sample_id)
        candidates.append(
            {
                "sample_id": sample_id,
                "seed": candidate_seed(42, problem_id, sample_id),
                "generated_token_ids": [sample_id + 1] * 512,
                "generated_text": f"answer {answer}",
                "extracted_answer": answer,
                "correct": answer == gold,
                "call_counts": {
                    "logical_denoising": 128,
                    "logical_scorer": 0,
                    "logical_total": 128,
                },
            }
        )
    return candidates


def _fake_problem(problem_id: int):
    return build_problem_record(
        global_problem_id=problem_id,
        question=f"question {problem_id}",
        gold_answer="7",
        full_gold_solution="work #### 7",
        prompt_text=f"prompt {problem_id}",
        prompt_token_ids=[1, 2, problem_id + 3],
        base_seed=42,
        num_candidates=8,
        candidates=_fake_candidates(problem_id),
        denoise_config=fixed_denoise_config(),
        execution={
            "new_candidates": 8,
            "logical_denoising": 1024,
            "physical_denoising_forwards": 128,
        },
        wall_time_seconds=1.0,
    )


def _fake_generation_manifest():
    return {
        "git": {"commit": "abc", "dirty": False},
        "artifacts": {
            "dream_backbone": {
                "path": "/model",
                "sha256": "model-sha",
            }
        },
        "config": {"sampler": sampler_manifest_config()},
        "extra": {"prompt_mode": "raw_8shot_no_chat_template"},
    }


def test_candidate_shard_merge_preserves_complete_schema_and_call_counts():
    shards = []
    for shard_id in range(2):
        record = _fake_problem(shard_id)
        shards.append(
            {
                "schema_version": POOL_SCHEMA_VERSION,
                "task": "gsm8k",
                "split": "test",
                "method": "unguided_dream_candidate_pool",
                "protocol_id": POOL_PROTOCOL_ID,
                "seed": 42,
                "num_candidates": 8,
                "shard_id": shard_id,
                "num_shards": 2,
                "complete": True,
                "manifest": _fake_generation_manifest(),
                "per_problem": [record],
            }
        )
    merged = merge_shard_payloads(
        shards,
        split="test",
        seed=42,
        num_candidates=8,
        expected_problems=2,
        formal_run=False,
    )
    assert merged["complete"]
    assert [item["global_problem_id"] for item in merged["per_problem"]] == [
        0,
        1,
    ]
    assert merged["summary"]["total_candidates"] == 16
    assert merged["summary"]["logical_call_totals"]["logical_denoising"] == 2 * 8 * 128
    validated = validate_pool_payload(merged, formal_run=False)
    assert [item["global_problem_id"] for item in validated] == [0, 1]


def test_cpu_scorer_selection_and_analysis_summary():
    source = {
        "global_problem_id": 4,
        "question": "q",
        "gold_answer": "9",
        "prompt_sha256": "prompt-sha",
        "prompt_token_ids": [1, 2],
        "call_counts": {"logical_denoising": 384},
        "candidates": [
            {
                "sample_id": 0,
                "seed": 10,
                "generated_token_ids": [1, 1],
                "extracted_answer": None,
                "correct": False,
                "call_counts": {"logical_denoising": 128},
            },
            {
                "sample_id": 1,
                "seed": 11,
                "generated_token_ids": [3, 4],
                "extracted_answer": "9",
                "correct": True,
                "call_counts": {"logical_denoising": 128},
            },
            {
                "sample_id": 2,
                "seed": 12,
                "generated_token_ids": [2, 2],
                "extracted_answer": "8",
                "correct": False,
                "call_counts": {"logical_denoising": 128},
            },
        ],
    }
    prm_scores = score_problem_candidates(
        model=TinyPRM(),
        scorer="prm",
        record=source,
        device="cpu",
        batch_size=2,
    )
    orm_scores = score_problem_candidates(
        model=TinyORM(),
        scorer="orm",
        record=source,
        device="cpu",
        batch_size=2,
    )
    assert prm_scores["call_counts"] == {
        "logical_scorer": 3,
        "physical_scorer_forwards": 2,
    }
    assert rerank_readout(source["candidates"], prm_scores)["selected_sample_id"] == 1

    # Explicitly exercise exact-score tie handling.
    tied = {
        "sample_ids": [0, 1, 2],
        "scores": [5.0, 5.0, 1.0],
        "call_counts": {
            "logical_scorer": 3,
            "physical_scorer_forwards": 1,
        },
    }
    assert rerank_readout(source["candidates"], tied)["selected_sample_id"] == 0

    result = build_analysis_problem(source, {"prm": prm_scores, "orm": orm_scores})
    methods = parse_methods("prm,orm")
    summary = summarize_analysis([result], methods)
    assert methods == ("majority", "oracle", "prm", "orm")
    assert summary["accuracy"]["oracle"] == 1.0
    assert summary["accuracy"]["prm"] == 1.0
    assert summary["accuracy"]["orm"] == 1.0
    assert result["candidates"][1]["scores"].keys() == {"prm", "orm"}


def test_analysis_shard_schema_merges_without_candidate_duplication():
    source_pool = {
        "protocol_id": POOL_PROTOCOL_ID,
        "split": "test",
        "seed": 42,
        "num_candidates": 8,
    }
    methods = ("majority", "oracle")
    source_records = [_fake_problem(0), _fake_problem(1)]
    analysis_records = [build_analysis_problem(record, {}) for record in source_records]
    shards = []
    for shard_id, record in enumerate(analysis_records):
        identity = {
            "protocol_id": "gsm8k_shared_candidate_rerank_v1",
            "num_shards": 2,
            "shard_id": shard_id,
            "selected_problem_ids": [shard_id],
        }
        shards.append(
            make_analysis_payload(
                source_pool=source_pool,
                methods=methods,
                shard_id=shard_id,
                num_shards=2,
                selected_problem_ids=[shard_id],
                run_identity=identity,
                manifest={"git": {"commit": "abc", "dirty": False}},
                scorer_state={},
                records=[record],
                complete=True,
            )
        )
    merged = merge_analysis_payloads(
        shards,
        source_pool=source_pool,
        methods=methods,
        expected_problem_ids=[0, 1],
        formal_run=False,
    )
    assert merged["complete"]
    assert merged["summary"]["num_problems"] == 2
    assert len(merged["per_problem"]) == 2
    assert all(len(record["candidates"]) == 8 for record in merged["per_problem"])


def test_submitted_orm_gate_locks_provenance_and_equivalence():
    adapter_sha = SUBMITTED_ORM_EXPECTATIONS["adapter_sha256"]
    source_sha = SUBMITTED_ORM_EXPECTATIONS["source_checkpoint_sha256"]
    config = {
        "trajectory_dir": "data/prm_trajectories/gsm8k_combined",
        "batch_size": 8,
        "grad_accum": 4,
        "lr": 2e-5,
        "max_steps": 500,
        "seed": 42,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
    }
    report = {
        "checkpoint_format": "safetensors_adapter",
        "model_type": "OutcomeRewardModel",
        "audited_loader": "load_state_dict_audited",
        "unexpected_keys": [],
        "config": config,
    }
    provenance = {
        "source_checkpoint_sha256": source_sha,
        "adapter_sha256": adapter_sha,
        "checkpoint_step": 500,
        "training_config": config,
        "source_summary": {
            "train_samples": 53190,
            "val_samples": 5926,
            "best_val_acc": 0.8904826189672629,
            "total_steps": 500,
        },
    }
    equivalence = {
        "all_passed": True,
        "source_checkpoint_sha256": source_sha,
        "adapter_sha256": adapter_sha,
        "num_states": 4,
        "max_abs_logit_difference": 0.0,
        "architecture_match": True,
        "architecture_config_match": True,
        "full_load_audit": {"passed": True},
        "adapter_load_audit": {
            "passed": True,
            "missing_trainable_keys": [],
        },
    }
    gate = validate_submitted_orm_gate(
        checkpoint_report=report,
        adapter_sha256=adapter_sha,
        source_checkpoint_sha256=source_sha,
        provenance=provenance,
        equivalence=equivalence,
    )
    assert gate["all_passed"] is True
    assert gate["max_abs_logit_difference"] == 0.0

    broken = dict(equivalence)
    broken["adapter_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="adapter SHA256"):
        validate_submitted_orm_gate(
            checkpoint_report=report,
            adapter_sha256=adapter_sha,
            source_checkpoint_sha256=source_sha,
            provenance=provenance,
            equivalence=broken,
        )

    forged_adapter_sha = "a" * 64
    forged_provenance = dict(provenance, adapter_sha256=forged_adapter_sha)
    forged_equivalence = dict(equivalence, adapter_sha256=forged_adapter_sha)
    with pytest.raises(ValueError, match="audited submitted checkpoint adapter"):
        validate_submitted_orm_gate(
            checkpoint_report=report,
            adapter_sha256=forged_adapter_sha,
            source_checkpoint_sha256=source_sha,
            provenance=forged_provenance,
            equivalence=forged_equivalence,
        )

    forged_source_sha = "b" * 64
    forged_provenance = dict(
        provenance,
        source_checkpoint_sha256=forged_source_sha,
    )
    forged_equivalence = dict(
        equivalence,
        source_checkpoint_sha256=forged_source_sha,
    )
    with pytest.raises(ValueError, match="audited submitted checkpoint"):
        validate_submitted_orm_gate(
            checkpoint_report=report,
            adapter_sha256=adapter_sha,
            source_checkpoint_sha256=forged_source_sha,
            provenance=forged_provenance,
            equivalence=forged_equivalence,
        )
