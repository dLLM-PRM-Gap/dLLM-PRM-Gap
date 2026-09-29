import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.prm.checkpointing import load_state_dict_audited, read_orm_checkpoint
from src.prm.generate_math_train_trajectories import (
    FORMAL_BASE_SEED,
    FORMAL_SNAPSHOT_EVERY,
    FORMAL_TRAJECTORY_BATCH_SIZE,
    FORMAL_TRAJECTORIES_PER_PROBLEM,
    TRAJECTORY_PROTOCOL_ID,
    expected_snapshot_steps,
    freeze_problem_split,
    trajectory_seed,
)
from src.prm.math_candidate_pool import (
    fixed_denoise_config,
    sampler_manifest_config,
)
from src.prm.math_protocol import (
    ANTLR_RUNTIME_VERSION,
    MATH500_QUESTION_LIST_SHA256,
    MATH500_REVISION,
    MATH_GRADER_PROTOCOL_ID,
    MATH_TRAIN_EXPECTED_PROBLEMS,
    MATH_TRAIN_QUESTION_LIST_SHA256,
    MATH_TRAIN_REVISION,
    MATH_VERIFY_VERSION,
)
from src.prm.protocol import sha256_text
from src.prm.train_orm import (
    _validate_trajectory_manifest,
    build_parser,
    describe_trajectory_source,
    enforce_output_contract,
    load_trajectories,
    load_problem_split,
    state_dict_sha256,
    trainable_state_dict,
    validate_args,
    validate_formal_math_trajectories,
    validate_formal_problem_split,
    validate_resume_config,
    validate_resume_training_manifest,
    resolve_trajectory_file,
)


class TinyORM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.reward_head = nn.Linear(2, 1)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False


def test_legacy_warmup_fraction_cli_remains_accepted_and_recorded():
    args = build_parser().parse_args(
        [
            "--trajectory_dir",
            "/tmp/trajectories",
            "--warmup_fraction",
            "0.25",
        ]
    )
    assert args.warmup_fraction == 0.25


def test_legacy_shard_directory_fallback_is_sorted_and_content_addressed(tmp_path):
    torch.save([{"problem_id": "later"}], tmp_path / "shard_001.pt")
    torch.save([{"problem_id": "first"}], tmp_path / "shard_000.pt")
    args = SimpleNamespace(
        trajectory_file=None,
        trajectory_dir=str(tmp_path),
        formal_run=False,
    )
    source = resolve_trajectory_file(args)
    assert source == tmp_path.resolve()
    assert [item["problem_id"] for item in load_trajectories(source)] == [
        "first",
        "later",
    ]
    descriptor = describe_trajectory_source(source)
    assert descriptor["kind"] == "shard-set"
    assert len(descriptor["sha256"]) == 64
    assert [item["name"] for item in descriptor["files"]] == [
        "shard_000.pt",
        "shard_001.pt",
    ]

    args.formal_run = True
    with pytest.raises(FileNotFoundError, match="merged trajectory"):
        resolve_trajectory_file(args)


def test_trainable_only_orm_checkpoint_roundtrip_is_audited(tmp_path):
    model = TinyORM()
    state = trainable_state_dict(model)
    assert set(state) == {"reward_head.weight", "reward_head.bias"}
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "trainable_state_dict": state,
            "config": {"lora_r": 16},
        },
        checkpoint,
    )
    loaded, config, checkpoint_format, _ = read_orm_checkpoint(checkpoint)
    target = TinyORM()
    missing, unexpected = load_state_dict_audited(
        target,
        loaded,
        adapter_only=True,
    )
    assert missing
    assert not unexpected
    assert config["lora_r"] == 16
    assert checkpoint_format == "pt_trainable"
    assert state_dict_sha256(state) == state_dict_sha256(loaded)


def test_frozen_problem_split_must_be_disjoint_and_cover_all_rows(tmp_path):
    trajectories = [
        {"problem_id": "p0", "is_correct": True},
        {"problem_id": "p0", "is_correct": False},
        {"problem_id": "p1", "is_correct": True},
    ]
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "fitting_problem_ids": ["p0"],
                "validation_problem_ids": ["p1"],
            }
        )
    )
    fitting, validation, _ = load_problem_split(path, trajectories)
    assert {item["problem_id"] for item in fitting} == {"p0"}
    assert {item["problem_id"] for item in validation} == {"p1"}

    path.write_text(
        json.dumps(
            {
                "fitting_problem_ids": ["p0"],
                "validation_problem_ids": ["p0", "p1"],
            }
        )
    )
    with pytest.raises(ValueError, match="overlaps"):
        load_problem_split(path, trajectories)


def test_formal_problem_split_must_match_frozen_algorithm():
    trajectories = [
        {"problem_id": f"p{problem_id}"} for problem_id in range(20) for _ in range(2)
    ]
    split = freeze_problem_split([f"p{problem_id}" for problem_id in range(20)])
    audit = validate_formal_problem_split(split, trajectories)
    assert audit["num_validation_problems"] == 2

    changed = dict(split)
    changed["fitting_problem_ids_sha256"] = "tampered"
    with pytest.raises(ValueError, match="frozen protocol"):
        validate_formal_problem_split(changed, trajectories)


def _formal_math_record(problem_ordinal, trajectory_id):
    problem_id = f"algebra:train:{problem_ordinal}"
    config = fixed_denoise_config()
    steps = expected_snapshot_steps(
        config,
        snapshot_every=FORMAL_SNAPSHOT_EVERY,
    )
    return {
        "schema_version": 1,
        "protocol_id": TRAJECTORY_PROTOCOL_ID,
        "task": "math_train",
        "problem_id": problem_id,
        "global_problem_id": problem_id,
        "problem_ordinal": problem_ordinal,
        "problem_sha256": f"digest-{problem_ordinal}",
        "trajectory_id": trajectory_id,
        "seed": trajectory_seed(
            FORMAL_BASE_SEED,
            problem_id,
            trajectory_id,
        ),
        "prompt_ids": torch.tensor([1, 2], dtype=torch.int32),
        "prompt_sha256": sha256_text(json.dumps([1, 2], separators=(",", ":"))),
        "gen_snapshots": torch.zeros(
            (len(steps), config.gen_length),
            dtype=torch.int32,
        ),
        "snapshot_steps": steps,
        "mask_ratios": [0.0] * len(steps),
        "generated_token_ids": [0] * config.gen_length,
        "answer_target": "7",
        "answer_extracted": "7",
        "extraction": {"answer": "7", "status": "ok"},
        "grade": {"correct": trajectory_id == 0, "status": "ok"},
        "is_correct": trajectory_id == 0,
        "call_counts": {"logical_denoising": config.steps},
    }


def test_formal_math_trajectory_audit_checks_exact_pairs_and_ordinals():
    trajectories = [
        _formal_math_record(problem_ordinal, trajectory_id)
        for problem_ordinal in range(2)
        for trajectory_id in range(2)
    ]
    audit = validate_formal_math_trajectories(
        trajectories,
        expected_problem_count=2,
        trajectories_per_problem=2,
    )
    assert audit["num_trajectories"] == 4
    assert audit["positive_trajectories"] == 2

    duplicated = [*trajectories[:-1], trajectories[0]]
    with pytest.raises(ValueError, match="duplicate"):
        validate_formal_math_trajectories(
            duplicated,
            expected_problem_count=2,
            trajectories_per_problem=2,
        )


def test_resume_config_rejects_data_or_optimizer_protocol_change():
    saved = {
        "trajectory_file": "/data/a.pt",
        "trajectory_sha256": "a",
        "batch_size": 4,
        "grad_accum": 8,
        "lr": 2e-5,
    }
    current = dict(saved)
    current["trajectory_sha256"] = "b"
    with pytest.raises(RuntimeError, match="trajectory_sha256"):
        validate_resume_config(saved, current)


def test_resume_manifest_locks_source_artifacts_and_protocol():
    saved = {
        "created_at_utc": "first",
        "git": {"commit": "abc", "dirty": False},
        "config": {"training_protocol_id": "v1"},
        "artifacts": {"trajectory_data": {"sha256": "data"}},
    }
    current = {**saved, "created_at_utc": "later"}
    validate_resume_training_manifest(saved, current)
    current = json.loads(json.dumps(current))
    current["git"]["commit"] = "different"
    with pytest.raises(RuntimeError, match="provenance"):
        validate_resume_training_manifest(saved, current)


def test_output_contract_rejects_stale_new_run_and_incomplete_resume(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"stale")
    with pytest.raises(FileExistsError, match="prior run artifacts"):
        enforce_output_contract(tmp_path, resume=False)
    with pytest.raises(FileNotFoundError, match="resume output is incomplete"):
        enforce_output_contract(tmp_path, resume=True)


def test_formal_args_require_all_artifact_attestations():
    args = SimpleNamespace(
        epochs=3,
        batch_size=4,
        grad_accum=8,
        lr=2e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        save_every_steps=500,
        eval_every_steps=100,
        log_every_steps=10,
        max_steps=0,
        seed=42,
        formal_run=True,
        trajectory_sha256=None,
        trajectory_manifest=None,
        split_manifest=None,
        split_manifest_sha256=None,
        model_sha256=None,
        model_hash_attestation=None,
        wandb=False,
        wandb_run_id=None,
    )
    with pytest.raises(ValueError, match="trajectory_sha256"):
        validate_args(args)

    wrong_seed = SimpleNamespace(**vars(args))
    wrong_seed.seed = 44
    with pytest.raises(ValueError, match="configuration mismatch"):
        validate_args(wrong_seed)

    wrong_schedule = SimpleNamespace(**vars(args))
    wrong_schedule.eval_every_steps = 99
    with pytest.raises(ValueError, match="eval_every_steps"):
        validate_args(wrong_schedule)


def test_formal_trajectory_manifest_checks_grader_and_overlap(tmp_path):
    trajectory_file = tmp_path / "trajectories_all.pt"
    trajectory_file.write_bytes(b"artifact")
    payload = {
        "complete": True,
        "task": "math_train",
        "protocol_id": TRAJECTORY_PROTOCOL_ID,
        "num_problems": MATH_TRAIN_EXPECTED_PROBLEMS,
        "num_trajectories": (
            MATH_TRAIN_EXPECTED_PROBLEMS * FORMAL_TRAJECTORIES_PER_PROBLEM
        ),
        "trajectories_per_problem": FORMAL_TRAJECTORIES_PER_PROBLEM,
        "trajectory_artifact": {
            "path": str(trajectory_file),
            "sha256": "sha",
        },
        "shard_manifests": [
            {
                "git": {"commit": "abc", "dirty": False},
                "artifacts": {
                    "dream_backbone": {"sha256": "model-sha"},
                    "dream_backbone_hash_attestation": {"sha256": "attestation-sha"},
                },
                "config": {
                    "protocol_id": TRAJECTORY_PROTOCOL_ID,
                    "base_seed": FORMAL_BASE_SEED,
                    "num_trajectories": FORMAL_TRAJECTORIES_PER_PROBLEM,
                    "trajectory_batch_size": FORMAL_TRAJECTORY_BATCH_SIZE,
                    "snapshot_every": FORMAL_SNAPSHOT_EVERY,
                    "snapshot_steps": expected_snapshot_steps(
                        fixed_denoise_config(),
                        snapshot_every=FORMAL_SNAPSHOT_EVERY,
                    ),
                    "sampler": sampler_manifest_config(),
                },
                "prompt_template_sha256": "prompt-sha",
                "extraction_callable_sha256": "extract-sha",
                "extra": {
                    "math_train_dataset": {
                        "revision": MATH_TRAIN_REVISION,
                        "question_list_sha256": (MATH_TRAIN_QUESTION_LIST_SHA256),
                        "expected_question_list_sha256": (
                            MATH_TRAIN_QUESTION_LIST_SHA256
                        ),
                    },
                    "math500_dataset": {
                        "revision": MATH500_REVISION,
                        "question_list_sha256": (MATH500_QUESTION_LIST_SHA256),
                        "expected_question_list_sha256": (MATH500_QUESTION_LIST_SHA256),
                    },
                    "math_grader": {
                        "protocol_id": MATH_GRADER_PROTOCOL_ID,
                        "formal_run": True,
                        "dependency_versions": {
                            "math-verify": MATH_VERIFY_VERSION,
                            "antlr4-python3-runtime": ANTLR_RUNTIME_VERSION,
                        },
                        "backend_grade_callable_sha256": "a" * 64,
                        "backend_equivalence_callable_sha256": "b" * 64,
                        "backend_extraction_callable_sha256": "c" * 64,
                    },
                    "train_eval_overlap": {"passed": True},
                },
            }
        ],
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload))
    loaded = _validate_trajectory_manifest(
        manifest,
        trajectory_file=trajectory_file,
        trajectory_sha256="sha",
        formal_run=True,
    )
    assert loaded["task"] == "math_train"

    payload["shard_manifests"][0]["extra"]["math_grader"]["dependency_versions"][
        "math-verify"
    ] = "moving"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="wrong math-verify"):
        _validate_trajectory_manifest(
            manifest,
            trajectory_file=trajectory_file,
            trajectory_sha256="sha",
            formal_run=True,
        )
