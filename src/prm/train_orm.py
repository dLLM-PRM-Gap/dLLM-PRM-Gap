"""Train a deterministic final-state Outcome Reward Model.

The hardened trainer stores only trainable LoRA/reward-head parameters, uses a
frozen problem-disjoint split, selects checkpoints by validation BCE, and
persists optimizer/scheduler/RNG plus the exact next dataloader position.
Formal runs additionally attest the Dream backbone and merged MATH trajectory
artifact before allocating the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


RELEASE_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(RELEASE_ROOT / "src"))

if __package__ in {None, ""}:
    from prm.checkpointing import load_state_dict_audited
    from prm.dataset import ORMDataset, collate_fn
    from prm.generate_math_train_trajectories import (
        FORMAL_BASE_SEED,
        FORMAL_SNAPSHOT_EVERY,
        FORMAL_SPLIT_SEED,
        FORMAL_TRAJECTORY_BATCH_SIZE,
        FORMAL_TRAJECTORIES_PER_PROBLEM,
        FORMAL_VAL_FRACTION,
        TRAJECTORY_PROTOCOL_ID,
        expected_snapshot_steps,
        freeze_problem_split,
        validate_trajectory_record,
        validate_trajectory_source_identity,
    )
    from prm.math_candidate_pool import (
        fixed_denoise_config,
        sampler_manifest_config,
    )
    from prm.math_protocol import (
        ANTLR_RUNTIME_VERSION,
        MATH500_REVISION,
        MATH500_QUESTION_LIST_SHA256,
        MATH_GRADER_PROTOCOL_ID,
        MATH_TRAIN_EXPECTED_PROBLEMS,
        MATH_TRAIN_QUESTION_LIST_SHA256,
        MATH_TRAIN_REVISION,
        MATH_VERIFY_VERSION,
        assert_no_problem_overlap,
        audit_problem_overlap,
        load_math500_examples,
        load_math_train_examples,
    )
    from prm.model import OutcomeRewardModel
    from prm.protocol import (
        artifact_descriptor,
        atomic_write_json,
        build_protocol_manifest,
        sha256_file,
        validate_artifact_attestation,
    )
else:
    from .checkpointing import load_state_dict_audited
    from .dataset import ORMDataset, collate_fn
    from .generate_math_train_trajectories import (
        FORMAL_BASE_SEED,
        FORMAL_SNAPSHOT_EVERY,
        FORMAL_SPLIT_SEED,
        FORMAL_TRAJECTORY_BATCH_SIZE,
        FORMAL_TRAJECTORIES_PER_PROBLEM,
        FORMAL_VAL_FRACTION,
        TRAJECTORY_PROTOCOL_ID,
        expected_snapshot_steps,
        freeze_problem_split,
        validate_trajectory_record,
        validate_trajectory_source_identity,
    )
    from .math_candidate_pool import (
        fixed_denoise_config,
        sampler_manifest_config,
    )
    from .math_protocol import (
        ANTLR_RUNTIME_VERSION,
        MATH500_REVISION,
        MATH500_QUESTION_LIST_SHA256,
        MATH_GRADER_PROTOCOL_ID,
        MATH_TRAIN_EXPECTED_PROBLEMS,
        MATH_TRAIN_QUESTION_LIST_SHA256,
        MATH_TRAIN_REVISION,
        MATH_VERIFY_VERSION,
        assert_no_problem_overlap,
        audit_problem_overlap,
        load_math500_examples,
        load_math_train_examples,
    )
    from .model import OutcomeRewardModel
    from .protocol import (
        artifact_descriptor,
        atomic_write_json,
        build_protocol_manifest,
        sha256_file,
        validate_artifact_attestation,
    )


LOGGER = logging.getLogger("train_orm")
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

FORMAL_MATH_ORM_SEEDS = (42, 43)
ORM_TRAINING_PROTOCOL_ID = "math_outcome_reward_model_training_v1"
FORMAL_MATH_ORM_CONFIG = {
    "epochs": 3,
    "batch_size": 4,
    "grad_accum": 8,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "save_every_steps": 500,
    "eval_every_steps": 100,
    "log_every_steps": 10,
    "max_steps": 0,
}


RESUME_CRITICAL_CONFIG = (
    "trajectory_dir",
    "trajectory_file",
    "trajectory_sha256",
    "trajectory_manifest",
    "split_manifest",
    "split_manifest_sha256",
    "model_path",
    "model_sha256",
    "epochs",
    "batch_size",
    "grad_accum",
    "lr",
    "weight_decay",
    "warmup_fraction",
    "max_grad_norm",
    "max_steps",
    "seed",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "save_every_steps",
    "eval_every_steps",
    "log_every_steps",
    "expected_init_sha256",
    "formal_run",
    "model_hash_attestation",
    "wandb",
    "wandb_project",
    "wandb_entity",
    "run_name",
    "wandb_run_id",
    "training_protocol_id",
)


def atomic_torch_save(payload: Any, destination: str | os.PathLike[str]) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def validate_resume_config(
    saved: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None:
    mismatches = {
        key: {"saved": saved.get(key), "current": current.get(key)}
        for key in RESUME_CRITICAL_CONFIG
        if saved.get(key) != current.get(key)
    }
    if mismatches:
        raise RuntimeError(
            "Resume configuration differs from checkpoint: "
            + json.dumps(mismatches, sort_keys=True)
        )


def enforce_output_contract(output_dir: Path, *, resume: bool) -> None:
    """Prevent a new run from inheriting stale artifacts and gate resume."""
    required_for_resume = {
        output_dir / "config.json",
        output_dir / "training_manifest.json",
        output_dir / "latest.pt",
    }
    material_artifacts = {
        *required_for_resume,
        output_dir / "best.pt",
        output_dir / "final.pt",
        output_dir / "summary.json",
        output_dir / "val_final.json",
    }
    material_artifacts.update(output_dir.glob("step_*.pt"))
    if resume:
        missing = sorted(
            str(path) for path in required_for_resume if not path.is_file()
        )
        if missing:
            raise FileNotFoundError(
                "resume output is incomplete; missing: " + ", ".join(missing)
            )
        return
    existing = sorted(str(path) for path in material_artifacts if path.exists())
    if existing:
        raise FileExistsError(
            "non-resume ORM output contains prior run artifacts: " + ", ".join(existing)
        )


def validate_resume_training_manifest(
    saved: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None:
    """Require the same source, artifacts, dependencies, and scientific config."""

    def identity(value: Mapping[str, Any]) -> dict[str, Any]:
        normalized = json.loads(json.dumps(dict(value), sort_keys=True))
        normalized.pop("created_at_utc", None)
        return normalized

    if identity(saved) != identity(current):
        raise RuntimeError(
            "resume training manifest differs from the original run provenance"
        )


def load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} must contain a JSON object")
    return dict(payload)


def resolve_trajectory_file(args: argparse.Namespace) -> Path:
    if args.trajectory_file:
        path = Path(args.trajectory_file).expanduser().resolve()
    else:
        directory = Path(args.trajectory_dir).expanduser().resolve()
        path = directory / "trajectories_all.pt"
        if not path.is_file():
            if args.formal_run:
                raise FileNotFoundError(
                    "formal ORM training requires the merged trajectory artifact: "
                    f"{path}"
                )
            shards = sorted(directory.glob("shard_*.pt"))
            if not shards:
                raise FileNotFoundError(
                    f"no trajectories_all.pt or shard_*.pt in {directory}"
                )
            return directory
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_trajectories(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    sources = sorted(resolved.glob("shard_*.pt")) if resolved.is_dir() else [resolved]
    if not sources:
        raise FileNotFoundError(f"no shard_*.pt in {resolved}")
    trajectories: list[dict[str, Any]] = []
    for source in sources:
        payload = torch.load(
            source,
            weights_only=False,
            map_location="cpu",
        )
        if isinstance(payload, Mapping) and isinstance(
            payload.get("trajectories"), list
        ):
            payload = payload["trajectories"]
        if not isinstance(payload, list):
            raise TypeError(f"trajectory artifact must contain a list: {source}")
        trajectories.extend(dict(item) for item in payload)
    if not trajectories:
        raise ValueError("trajectory artifact is empty")
    return trajectories


def describe_trajectory_source(path: Path) -> dict[str, Any]:
    """Content-address one merged file or a deterministic legacy shard set."""
    if path.is_file():
        return artifact_descriptor(path, require_hash=True)
    shard_files = sorted(path.glob("shard_*.pt"))
    if not shard_files:
        raise FileNotFoundError(f"no shard_*.pt in {path}")
    files = [
        {
            "name": shard.name,
            "size_bytes": shard.stat().st_size,
            "sha256": sha256_file(shard),
        }
        for shard in shard_files
    ]
    digest = hashlib.sha256()
    for record in files:
        encoded = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return {
        "path": str(path),
        "kind": "shard-set",
        "sha256": digest.hexdigest(),
        "sha256_scheme": "sorted-shard-name-size-and-file-sha256-v1",
        "files": files,
    }


def validate_formal_math_trajectories(
    trajectories: Sequence[Mapping[str, Any]],
    *,
    expected_problem_count: int = MATH_TRAIN_EXPECTED_PROBLEMS,
    trajectories_per_problem: int = FORMAL_TRAJECTORIES_PER_PROBLEM,
) -> dict[str, Any]:
    """Audit the complete formal MATH training artifact before model load."""
    expected_trajectory_count = int(expected_problem_count) * int(
        trajectories_per_problem
    )
    if len(trajectories) != expected_trajectory_count:
        raise ValueError(
            f"formal MATH artifact has {len(trajectories)} trajectories; "
            f"expected {expected_trajectory_count}"
        )

    trajectory_ids_by_problem: dict[str, set[int]] = {}
    ordinal_by_problem: dict[str, int] = {}
    problem_by_ordinal: dict[int, str] = {}
    seen_pairs: set[tuple[str, int]] = set()
    status_counts: dict[str, int] = {}
    positive_count = 0
    for record in trajectories:
        if record.get("schema_version") != 1:
            raise ValueError("formal MATH trajectory schema mismatch")
        if record.get("protocol_id") != TRAJECTORY_PROTOCOL_ID:
            raise ValueError("formal MATH trajectory protocol mismatch")
        if record.get("task") != "math_train":
            raise ValueError("formal ORM artifact contains a non-MATH row")
        problem_id = str(record["problem_id"])
        if str(record.get("global_problem_id")) != problem_id:
            raise ValueError("problem_id/global_problem_id mismatch")
        trajectory_id = int(record["trajectory_id"])
        pair = (problem_id, trajectory_id)
        if pair in seen_pairs:
            raise ValueError(f"duplicate formal trajectory pair: {pair}")
        seen_pairs.add(pair)
        validate_trajectory_record(
            record,
            base_seed=FORMAL_BASE_SEED,
            num_trajectories=trajectories_per_problem,
            snapshot_every=FORMAL_SNAPSHOT_EVERY,
        )

        ordinal = int(record["problem_ordinal"])
        previous_ordinal = ordinal_by_problem.setdefault(problem_id, ordinal)
        if previous_ordinal != ordinal:
            raise ValueError(f"problem {problem_id} has multiple ordinals")
        previous_problem = problem_by_ordinal.setdefault(ordinal, problem_id)
        if previous_problem != problem_id:
            raise ValueError(f"problem ordinal {ordinal} is reused")
        trajectory_ids_by_problem.setdefault(problem_id, set()).add(trajectory_id)
        grade = record["grade"]
        status = str(grade["status"])
        if status not in {"ok", "timeout", "error"}:
            raise ValueError(f"unsupported formal grade status: {status}")
        status_counts[status] = status_counts.get(status, 0) + 1
        positive_count += int(bool(record["is_correct"]))

    if len(trajectory_ids_by_problem) != expected_problem_count:
        raise ValueError(
            f"formal MATH artifact has {len(trajectory_ids_by_problem)} "
            f"problems; expected {expected_problem_count}"
        )
    expected_ids = set(range(trajectories_per_problem))
    incomplete = [
        problem_id
        for problem_id, observed_ids in trajectory_ids_by_problem.items()
        if observed_ids != expected_ids
    ]
    if incomplete:
        raise ValueError(
            "formal MATH problems lack the canonical trajectory IDs: "
            + ", ".join(sorted(incomplete)[:5])
        )
    if sorted(problem_by_ordinal) != list(range(expected_problem_count)):
        raise ValueError("formal MATH problem ordinals are not contiguous")
    return {
        "num_problems": len(trajectory_ids_by_problem),
        "num_trajectories": len(trajectories),
        "trajectories_per_problem": trajectories_per_problem,
        "positive_trajectories": positive_count,
        "grade_status_counts": status_counts,
    }


def load_problem_split(
    path: str | os.PathLike[str],
    trajectories: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        split = json.load(handle)
    fitting_ids = {str(value) for value in split["fitting_problem_ids"]}
    validation_ids = {str(value) for value in split["validation_problem_ids"]}
    if fitting_ids & validation_ids:
        raise ValueError("fitting/validation split overlaps")
    observed_ids = {str(item["problem_id"]) for item in trajectories}
    if fitting_ids | validation_ids != observed_ids:
        missing = observed_ids - fitting_ids - validation_ids
        extra = fitting_ids | validation_ids - observed_ids
        raise ValueError(
            f"split does not cover trajectory problems: "
            f"missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}"
        )
    fitting = [
        dict(item) for item in trajectories if str(item["problem_id"]) in fitting_ids
    ]
    validation = [
        dict(item) for item in trajectories if str(item["problem_id"]) in validation_ids
    ]
    if not fitting or not validation:
        raise ValueError("fitting and validation splits must both be nonempty")
    return fitting, validation, split


def validate_formal_problem_split(
    split: Mapping[str, Any],
    trajectories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require the exact prespecified problem-disjoint MATH split."""
    observed_problem_ids = sorted({str(item["problem_id"]) for item in trajectories})
    expected = freeze_problem_split(
        observed_problem_ids,
        seed=FORMAL_SPLIT_SEED,
        val_fraction=FORMAL_VAL_FRACTION,
    )
    checked_fields = (
        "schema_version",
        "split_protocol",
        "seed",
        "val_fraction",
        "num_fitting_problems",
        "num_validation_problems",
        "fitting_problem_ids",
        "validation_problem_ids",
        "fitting_problem_ids_sha256",
        "validation_problem_ids_sha256",
    )
    mismatches = {
        field: {
            "expected": expected.get(field),
            "actual": split.get(field),
        }
        for field in checked_fields
        if split.get(field) != expected.get(field)
    }
    if mismatches:
        raise ValueError(
            "formal problem split differs from the frozen protocol: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "split_protocol": expected["split_protocol"],
        "seed": expected["seed"],
        "val_fraction": expected["val_fraction"],
        "num_fitting_problems": expected["num_fitting_problems"],
        "num_validation_problems": expected["num_validation_problems"],
        "fitting_problem_ids_sha256": expected["fitting_problem_ids_sha256"],
        "validation_problem_ids_sha256": expected["validation_problem_ids_sha256"],
    }


def split_by_problem(
    trajectories: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    problem_ids = sorted({str(item["problem_id"]) for item in trajectories})
    random.Random(seed).shuffle(problem_ids)
    num_validation = min(
        max(len(problem_ids) - 1, 1),
        max(50, int(len(problem_ids) * val_fraction)),
    )
    validation_ids = set(problem_ids[:num_validation])
    fitting = [
        dict(item)
        for item in trajectories
        if str(item["problem_id"]) not in validation_ids
    ]
    validation = [
        dict(item) for item in trajectories if str(item["problem_id"]) in validation_ids
    ]
    split = {
        "split_protocol": "trainer_fallback_problem_shuffle_v1",
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "fitting_problem_ids": sorted({str(item["problem_id"]) for item in fitting}),
        "validation_problem_ids": sorted(validation_ids),
    }
    return fitting, validation, split


def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prompt_lens = batch["prompt_lens"].to(device)
            labels = batch["labels"].to(device)
            logits = model(
                input_ids,
                prompt_lens,
                attention_mask=attention_mask,
            )
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            total_loss += loss.item() * len(labels)
            predictions = (logits > 0).float()
            correct += int((predictions == labels).sum().item())
            total += len(labels)
    model.train()
    return total_loss / max(total, 1), correct / max(total, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trajectory_dir")
    source.add_argument("--trajectory_file")
    parser.add_argument("--trajectory_sha256", default=None)
    parser.add_argument("--trajectory_manifest", default=None)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--split_manifest_sha256", default=None)
    parser.add_argument(
        "--model_path",
        default="Dream-org/Dream-v0-Instruct-7B",
    )
    parser.add_argument("--model_sha256", default=None)
    parser.add_argument("--model_hash_attestation", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--output_dir", default="checkpoints/orm_gsm8k")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--warmup_fraction",
        type=float,
        default=0.1,
        help=(
            "Legacy CLI compatibility field recorded in config; the historical "
            "ORM trainer used CosineAnnealingLR without applying warmup."
        ),
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--save_every_steps", type=int, default=500)
    parser.add_argument("--eval_every_steps", type=int, default=100)
    parser.add_argument("--log_every_steps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected_init_sha256", default=None)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="dllm-prm-gap")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--run_name", default="orm_gsm8k")
    parser.add_argument("--wandb_run_id", default=None)
    parser.add_argument("--formal_run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_grad_norm": args.max_grad_norm,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "save_every_steps": args.save_every_steps,
        "eval_every_steps": args.eval_every_steps,
        "log_every_steps": args.log_every_steps,
    }
    invalid = {key: value for key, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"positive training arguments required: {invalid}")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("lora_dropout must be in [0, 1)")
    if args.max_steps < 0:
        raise ValueError("max_steps cannot be negative")
    if args.formal_run:
        mismatches = {
            key: {"expected": expected, "actual": getattr(args, key, None)}
            for key, expected in FORMAL_MATH_ORM_CONFIG.items()
            if getattr(args, key, None) != expected
        }
        if getattr(args, "seed", None) not in FORMAL_MATH_ORM_SEEDS:
            mismatches["seed"] = {
                "expected": list(FORMAL_MATH_ORM_SEEDS),
                "actual": getattr(args, "seed", None),
            }
        if mismatches:
            raise ValueError(
                "formal MATH ORM configuration mismatch: "
                + json.dumps(mismatches, sort_keys=True)
            )
        required = {
            "trajectory_sha256": args.trajectory_sha256,
            "trajectory_manifest": args.trajectory_manifest,
            "split_manifest": args.split_manifest,
            "split_manifest_sha256": args.split_manifest_sha256,
            "model_sha256": args.model_sha256,
            "model_hash_attestation": args.model_hash_attestation,
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ValueError("formal ORM training is missing: " + ", ".join(missing))
        if args.wandb and not args.wandb_run_id:
            raise ValueError("formal WandB training requires --wandb_run_id")


def _validate_trajectory_manifest(
    path: str | os.PathLike[str],
    *,
    trajectory_file: Path,
    trajectory_sha256: str,
    formal_run: bool,
) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("complete") is not True:
        raise ValueError("trajectory manifest is incomplete")
    artifact = manifest.get("trajectory_artifact", {})
    if artifact.get("sha256") != trajectory_sha256:
        raise ValueError("trajectory manifest hash differs from supplied hash")
    if Path(artifact.get("path", "")).name != trajectory_file.name:
        raise ValueError("trajectory manifest artifact name mismatch")
    if formal_run:
        if manifest.get("task") != "math_train":
            raise ValueError("formal ORM trajectories are not MATH train")
        expected_top_level = {
            "protocol_id": TRAJECTORY_PROTOCOL_ID,
            "num_problems": MATH_TRAIN_EXPECTED_PROBLEMS,
            "num_trajectories": (
                MATH_TRAIN_EXPECTED_PROBLEMS * FORMAL_TRAJECTORIES_PER_PROBLEM
            ),
            "trajectories_per_problem": FORMAL_TRAJECTORIES_PER_PROBLEM,
        }
        top_level_mismatches = {
            key: {"expected": value, "actual": manifest.get(key)}
            for key, value in expected_top_level.items()
            if manifest.get(key) != value
        }
        if top_level_mismatches:
            raise ValueError(
                "formal trajectory manifest shape/protocol mismatch: "
                + json.dumps(top_level_mismatches, sort_keys=True)
            )
        shard_manifests = manifest.get("shard_manifests")
        if not isinstance(shard_manifests, list) or not shard_manifests:
            raise ValueError("formal trajectory manifest lacks shard provenance")
        expected_sampler = sampler_manifest_config()
        expected_steps = expected_snapshot_steps(
            fixed_denoise_config(),
            snapshot_every=FORMAL_SNAPSHOT_EVERY,
        )
        source_signature = None
        for shard_manifest in shard_manifests:
            git = shard_manifest.get("git", {})
            if git.get("dirty") is not False or not git.get("commit"):
                raise ValueError(
                    "formal trajectory labels were generated from dirty source"
                )
            artifacts = shard_manifest.get("artifacts", {})
            model_artifact = artifacts.get("dream_backbone", {})
            if not model_artifact.get("sha256"):
                raise ValueError(
                    "formal trajectory manifest lacks a Dream backbone hash"
                )
            if not artifacts.get("dream_backbone_hash_attestation", {}).get("sha256"):
                raise ValueError(
                    "formal trajectory manifest lacks model-hash attestation"
                )
            shard_config = shard_manifest.get("config", {})
            expected_config = {
                "protocol_id": TRAJECTORY_PROTOCOL_ID,
                "base_seed": FORMAL_BASE_SEED,
                "num_trajectories": FORMAL_TRAJECTORIES_PER_PROBLEM,
                "trajectory_batch_size": FORMAL_TRAJECTORY_BATCH_SIZE,
                "snapshot_every": FORMAL_SNAPSHOT_EVERY,
                "snapshot_steps": expected_steps,
                "sampler": expected_sampler,
            }
            config_mismatches = {
                key: {
                    "expected": value,
                    "actual": shard_config.get(key),
                }
                for key, value in expected_config.items()
                if shard_config.get(key) != value
            }
            if config_mismatches:
                raise ValueError(
                    "formal trajectory shard protocol mismatch: "
                    + json.dumps(config_mismatches, sort_keys=True)
                )
            extra = shard_manifest.get("extra", {})
            train_dataset = extra.get("math_train_dataset", {})
            evaluation_dataset = extra.get("math500_dataset", {})
            if train_dataset.get("revision") != MATH_TRAIN_REVISION:
                raise ValueError("formal trajectories use the wrong MATH train")
            if evaluation_dataset.get("revision") != MATH500_REVISION:
                raise ValueError("formal trajectories use the wrong MATH500")
            if (
                train_dataset.get("question_list_sha256")
                != MATH_TRAIN_QUESTION_LIST_SHA256
                or train_dataset.get("expected_question_list_sha256")
                != MATH_TRAIN_QUESTION_LIST_SHA256
                or evaluation_dataset.get("question_list_sha256")
                != MATH500_QUESTION_LIST_SHA256
                or evaluation_dataset.get("expected_question_list_sha256")
                != MATH500_QUESTION_LIST_SHA256
            ):
                raise ValueError(
                    "formal trajectory dataset question-list lock mismatch"
                )
            grader = shard_manifest.get("extra", {}).get("math_grader", {})
            dependencies = grader.get("dependency_versions", {})
            if grader.get("protocol_id") != MATH_GRADER_PROTOCOL_ID:
                raise ValueError("formal trajectory labels use wrong grader")
            if grader.get("formal_run") is not True:
                raise ValueError("trajectory labels were not formally gated")
            if dependencies.get("math-verify") != MATH_VERIFY_VERSION:
                raise ValueError("formal trajectory labels use wrong math-verify")
            if dependencies.get("antlr4-python3-runtime") != (ANTLR_RUNTIME_VERSION):
                raise ValueError("formal trajectory labels use wrong antlr runtime")
            backend_hash_fields = (
                "backend_grade_callable_sha256",
                "backend_equivalence_callable_sha256",
                "backend_extraction_callable_sha256",
            )
            if any(not grader.get(field) for field in backend_hash_fields):
                raise ValueError("formal trajectory grader lacks backend source hashes")
            if extra.get("train_eval_overlap", {}).get("passed") is not True:
                raise ValueError("formal trajectory source failed overlap audit")
            signature = {
                "git_commit": git["commit"],
                "dream_backbone_sha256": model_artifact["sha256"],
                "prompt_template_sha256": shard_manifest.get("prompt_template_sha256"),
                "extraction_callable_sha256": shard_manifest.get(
                    "extraction_callable_sha256"
                ),
                "math_train_dataset": train_dataset,
                "math500_dataset": evaluation_dataset,
                "grader": grader,
            }
            if source_signature is None:
                source_signature = signature
            elif signature != source_signature:
                raise ValueError(
                    "formal trajectory shards have inconsistent provenance"
                )
    return manifest


def _build_training_manifest(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    model_artifact: Mapping[str, Any],
    model_attestation: Mapping[str, Any] | None,
    trajectory_artifact: Mapping[str, Any],
    trajectory_manifest_artifact: Mapping[str, Any] | None,
    split_artifact: Mapping[str, Any] | None,
    split: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts: dict[str, Mapping[str, Any]] = {
        "dream_backbone": model_artifact,
        "trajectory_data": trajectory_artifact,
    }
    if model_attestation is not None:
        artifacts["dream_backbone_hash_attestation"] = model_attestation
    if trajectory_manifest_artifact is not None:
        artifacts["trajectory_manifest"] = trajectory_manifest_artifact
    if split_artifact is not None:
        artifacts["problem_split"] = split_artifact
    return build_protocol_manifest(
        repo_root=RELEASE_ROOT,
        experiment="outcome_reward_model_training",
        config=dict(config),
        artifacts=artifacts,
        dependency_names=(
            "torch",
            "transformers",
            "peft",
            "safetensors",
            "numpy",
        ),
        extra={
            "training_protocol_id": ORM_TRAINING_PROTOCOL_ID,
            "checkpoint_format": "pt_trainable",
            "checkpoint_selection_metric": "validation_bce",
            "problem_split": {
                "split_protocol": split.get("split_protocol"),
                "seed": split.get("seed"),
                "val_fraction": split.get("val_fraction"),
                "num_fitting_problems": len(split.get("fitting_problem_ids", [])),
                "num_validation_problems": len(split.get("validation_problem_ids", [])),
            },
        },
    )


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    enforce_output_contract(output_dir, resume=args.resume)
    set_seed(args.seed)
    device = torch.device("cuda")

    trajectory_file = resolve_trajectory_file(args)
    trajectory_artifact = describe_trajectory_source(trajectory_file)
    actual_trajectory_sha256 = str(trajectory_artifact["sha256"])
    if args.trajectory_sha256 and actual_trajectory_sha256 != (args.trajectory_sha256):
        raise ValueError(
            "trajectory SHA-256 mismatch: "
            f"{actual_trajectory_sha256} != {args.trajectory_sha256}"
        )
    if trajectory_file.is_file():
        args.trajectory_file = str(trajectory_file)
    else:
        args.trajectory_file = None
        args.trajectory_dir = str(trajectory_file)
    args.trajectory_sha256 = actual_trajectory_sha256
    trajectories = load_trajectories(trajectory_file)
    formal_trajectory_audit = None
    if args.formal_run:
        formal_trajectory_audit = validate_formal_math_trajectories(trajectories)
        canonical_train, _ = load_math_train_examples(
            formal_run=True,
            cache_dir=args.cache_dir,
        )
        canonical_evaluation, _ = load_math500_examples(
            formal_run=True,
            cache_dir=args.cache_dir,
        )
        assert_no_problem_overlap(
            audit_problem_overlap(canonical_train, canonical_evaluation)
        )
        canonical_by_ordinal = {
            int(item["problem_ordinal"]): item for item in canonical_train
        }
        for trajectory in trajectories:
            ordinal = int(trajectory["problem_ordinal"])
            if ordinal not in canonical_by_ordinal:
                raise ValueError("ORM trajectory ordinal absent from canonical train")
            validate_trajectory_source_identity(
                trajectory,
                canonical_by_ordinal[ordinal],
            )

    trajectory_manifest_payload = None
    trajectory_manifest_artifact = None
    if args.trajectory_manifest:
        if not trajectory_file.is_file():
            raise ValueError(
                "trajectory manifests require one merged trajectory artifact"
            )
        trajectory_manifest_payload = _validate_trajectory_manifest(
            args.trajectory_manifest,
            trajectory_file=trajectory_file,
            trajectory_sha256=actual_trajectory_sha256,
            formal_run=args.formal_run,
        )
        trajectory_manifest_artifact = artifact_descriptor(
            args.trajectory_manifest,
            require_hash=True,
        )

    split_artifact = None
    if args.split_manifest:
        actual_split_sha256 = sha256_file(args.split_manifest)
        if args.split_manifest_sha256 and actual_split_sha256 != (
            args.split_manifest_sha256
        ):
            raise ValueError("problem split SHA-256 mismatch")
        args.split_manifest_sha256 = actual_split_sha256
        fitting_trajectories, validation_trajectories, split = load_problem_split(
            args.split_manifest, trajectories
        )
        formal_split_audit = None
        if args.formal_run:
            formal_split_audit = validate_formal_problem_split(
                split,
                trajectories,
            )
        split_artifact = artifact_descriptor(
            args.split_manifest,
            sha256=actual_split_sha256,
            require_hash=True,
        )
    else:
        if args.formal_run:
            raise ValueError("formal ORM training requires frozen split manifest")
        fitting_trajectories, validation_trajectories, split = split_by_problem(
            trajectories,
            seed=args.seed,
        )
        formal_split_audit = None

    positive_fitting = sum(bool(item["is_correct"]) for item in fitting_trajectories)
    positive_validation = sum(
        bool(item["is_correct"]) for item in validation_trajectories
    )
    LOGGER.info(
        "Fitting: %d trajectories (%d positive, %.2f%%); "
        "validation: %d (%d positive, %.2f%%)",
        len(fitting_trajectories),
        positive_fitting,
        100 * positive_fitting / len(fitting_trajectories),
        len(validation_trajectories),
        positive_validation,
        100 * positive_validation / len(validation_trajectories),
    )
    fitting_dataset = ORMDataset(fitting_trajectories)
    validation_dataset = ORMDataset(validation_trajectories)
    collate = partial(collate_fn)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
    )

    def make_fitting_loader(epoch: int) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(args.seed + int(epoch))
        return DataLoader(
            fitting_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate,
            num_workers=4,
            pin_memory=True,
        )

    config = vars(args).copy()
    config["training_protocol_id"] = ORM_TRAINING_PROTOCOL_ID
    config["trajectory_manifest_validated"] = bool(trajectory_manifest_payload)
    config["fitting_problem_count"] = len(split.get("fitting_problem_ids", []))
    config["validation_problem_count"] = len(split.get("validation_problem_ids", []))
    config["formal_trajectory_audit"] = formal_trajectory_audit
    config["formal_split_audit"] = formal_split_audit
    model_artifact = artifact_descriptor(
        args.model_path,
        sha256=args.model_sha256,
        require_hash=args.formal_run,
        verify_supplied_hash=False,
    )
    model_attestation = None
    if args.formal_run:
        attestation_payload = validate_artifact_attestation(
            args.model_hash_attestation,
            artifact_path=args.model_path,
            expected_sha256=args.model_sha256,
        )
        model_attestation = attestation_payload["attestation"]
    training_manifest = _build_training_manifest(
        args=args,
        config={**config, "resume": False},
        model_artifact=model_artifact,
        model_attestation=model_attestation,
        trajectory_artifact=trajectory_artifact,
        trajectory_manifest_artifact=trajectory_manifest_artifact,
        split_artifact=split_artifact,
        split=split,
    )
    if args.formal_run:
        git = training_manifest.get("git", {})
        if not git.get("commit") or git.get("dirty") is not False:
            raise ValueError("formal ORM training requires clean committed source")
    manifest_path = output_dir / "training_manifest.json"
    config_path = output_dir / "config.json"
    if args.resume:
        existing_manifest = load_json_mapping(
            manifest_path,
            label="training manifest",
        )
        validate_resume_training_manifest(existing_manifest, training_manifest)
        training_manifest = existing_manifest
        validate_resume_config(
            load_json_mapping(config_path, label="training config"),
            config,
        )
    else:
        atomic_write_json(config_path, config)
        atomic_write_json(manifest_path, training_manifest)

    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            id=args.wandb_run_id,
            resume="must" if args.resume else None,
            config=config,
        )
        if args.wandb_run_id and run.id != args.wandb_run_id:
            raise RuntimeError("WandB returned a different run id")
        args.wandb_run_id = run.id
        LOGGER.info("WandB run: %s", run.id)

    from dream.modeling_dream import DreamModel

    LOGGER.info("Loading backbone from %s", args.model_path)
    backbone = DreamModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model = OutcomeRewardModel(
        backbone,
        hidden_size=backbone.config.hidden_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model = model.to(device)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    initial_state = trainable_state_dict(model)
    init_sha256 = state_dict_sha256(initial_state)
    if args.expected_init_sha256 and init_sha256 != args.expected_init_sha256:
        raise RuntimeError(
            f"initial trainable-state mismatch: {init_sha256} != "
            f"{args.expected_init_sha256}"
        )
    config["init_sha256"] = init_sha256
    config["wandb_run_id"] = args.wandb_run_id
    if args.resume:
        saved_config = load_json_mapping(config_path, label="training config")
        validate_resume_config(saved_config, config)
        if saved_config.get("init_sha256") != init_sha256:
            raise RuntimeError("resume config initialization hash mismatch")
    else:
        atomic_write_json(config_path, config)
    LOGGER.info(
        "Trainable parameters: %d; init SHA-256: %s",
        model.trainable_params(),
        init_sha256,
    )

    optimizer = AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    batches_per_epoch = len(make_fitting_loader(0))
    updates_per_epoch = batches_per_epoch // args.grad_accum
    if updates_per_epoch <= 0:
        raise ValueError("fitting split is smaller than one optimizer update")
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = updates_per_epoch * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    LOGGER.info(
        "Total steps: %d; effective batch: %d",
        total_steps,
        args.batch_size * args.grad_accum,
    )

    start_step = 0
    start_epoch = 0
    start_batch_index = 0
    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    if args.resume:
        checkpoint_path = output_dir / "latest.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"--resume requested but checkpoint is absent: {checkpoint_path}"
            )
        checkpoint = torch.load(
            checkpoint_path,
            weights_only=False,
            map_location="cpu",
        )
        validate_resume_config(checkpoint.get("config", {}), config)
        saved_state = checkpoint.get(
            "trainable_state_dict",
            checkpoint.get("model_state_dict"),
        )
        if not isinstance(saved_state, Mapping):
            raise KeyError("resume checkpoint has no model state")
        load_state_dict_audited(
            model,
            saved_state,
            adapter_only="trainable_state_dict" in checkpoint,
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = int(checkpoint["step"])
        start_epoch = int(checkpoint.get("next_epoch", checkpoint.get("epoch", 0)))
        start_batch_index = int(checkpoint.get("next_batch_idx", 0))
        best_validation_loss = float(checkpoint.get("best_val_loss", float("inf")))
        best_validation_accuracy = float(checkpoint.get("best_val_acc", 0.0))
        if checkpoint.get("init_sha256") != init_sha256:
            raise RuntimeError("resume checkpoint initialization hash mismatch")
        if "rng_state" not in checkpoint:
            raise RuntimeError("resume checkpoint lacks RNG state")
        restore_rng_state(checkpoint["rng_state"])
        LOGGER.info(
            "Resumed from step=%d epoch=%d batch=%d",
            start_step,
            start_epoch,
            start_batch_index,
        )

    model.train()
    global_step = start_step
    accumulated_loss = 0.0
    optimizer.zero_grad()
    training_started = time.time()
    epoch = start_epoch
    while global_step < total_steps:
        fitting_loader = make_fitting_loader(epoch)
        resume_batch_index = start_batch_index if epoch == start_epoch else 0
        if resume_batch_index % args.grad_accum != 0:
            raise RuntimeError(
                f"resume batch must lie on optimizer boundary, got {resume_batch_index}"
            )
        optimizer.zero_grad()
        for batch_index, batch in enumerate(fitting_loader):
            if batch_index < resume_batch_index:
                continue
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prompt_lens = batch["prompt_lens"].to(device)
            labels = batch["labels"].to(device)
            logits = model(
                input_ids,
                prompt_lens,
                attention_mask=attention_mask,
            )
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            accumulated_loss += float(scaled_loss.item())

            if (batch_index + 1) % args.grad_accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_(
                trainable,
                args.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            next_epoch = epoch
            next_batch_index = batch_index + 1
            if next_batch_index >= len(fitting_loader):
                next_epoch = epoch + 1
                next_batch_index = 0

            if global_step % args.log_every_steps == 0:
                learning_rate = scheduler.get_last_lr()[0]
                LOGGER.info(
                    "step=%d/%d loss=%.6f lr=%.3e elapsed=%.1fmin",
                    global_step,
                    total_steps,
                    accumulated_loss,
                    learning_rate,
                    (time.time() - training_started) / 60,
                )
                if args.wandb:
                    wandb.log(
                        {
                            "train/loss": accumulated_loss,
                            "train/lr": learning_rate,
                        },
                        step=global_step,
                    )
                accumulated_loss = 0.0

            if global_step % args.eval_every_steps == 0:
                validation_loss, validation_accuracy = evaluate(
                    model,
                    validation_loader,
                    device,
                )
                LOGGER.info(
                    "validation step=%d loss=%.6f accuracy=%.2f%%",
                    global_step,
                    validation_loss,
                    100 * validation_accuracy,
                )
                if args.wandb:
                    wandb.log(
                        {
                            "val/loss": validation_loss,
                            "val/accuracy": validation_accuracy,
                        },
                        step=global_step,
                    )
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_validation_accuracy = validation_accuracy
                    atomic_torch_save(
                        {
                            "step": global_step,
                            "trainable_state_dict": trainable_state_dict(model),
                            "best_val_loss": best_validation_loss,
                            "best_val_acc": best_validation_accuracy,
                            "init_sha256": init_sha256,
                            "config": config,
                        },
                        output_dir / "best.pt",
                    )
                    LOGGER.info(
                        "New best validation BCE: %.6f",
                        best_validation_loss,
                    )

            if global_step % args.save_every_steps == 0:
                checkpoint = {
                    "step": global_step,
                    "epoch": epoch,
                    "next_epoch": next_epoch,
                    "next_batch_idx": next_batch_index,
                    "trainable_state_dict": trainable_state_dict(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_validation_loss,
                    "best_val_acc": best_validation_accuracy,
                    "init_sha256": init_sha256,
                    "rng_state": capture_rng_state(),
                    "wandb_run_id": args.wandb_run_id,
                    "config": config,
                }
                atomic_torch_save(checkpoint, output_dir / "latest.pt")
                LOGGER.info("Checkpoint saved at step %d", global_step)
            if global_step >= total_steps:
                break
        epoch += 1
        start_batch_index = 0

    final_validation_loss, final_validation_accuracy = evaluate(
        model,
        validation_loader,
        device,
    )
    # Persist final validation before final checkpoints so buffered logs are
    # never the only source for the last metric.
    atomic_write_json(
        output_dir / "val_final.json",
        {
            "bce": final_validation_loss,
            "accuracy": final_validation_accuracy,
            "checkpoint_selection_metric": "bce",
            "step": global_step,
        },
    )
    LOGGER.info(
        "Final validation: loss=%.6f accuracy=%.2f%%",
        final_validation_loss,
        100 * final_validation_accuracy,
    )
    if final_validation_loss < best_validation_loss:
        best_validation_loss = final_validation_loss
        best_validation_accuracy = final_validation_accuracy
        atomic_torch_save(
            {
                "step": global_step,
                "trainable_state_dict": trainable_state_dict(model),
                "best_val_loss": best_validation_loss,
                "best_val_acc": best_validation_accuracy,
                "init_sha256": init_sha256,
                "config": config,
            },
            output_dir / "best.pt",
        )
    final_state = trainable_state_dict(model)
    atomic_torch_save(
        {
            "step": global_step,
            "trainable_state_dict": final_state,
            "best_val_loss": best_validation_loss,
            "best_val_acc": best_validation_accuracy,
            "final_val_loss": final_validation_loss,
            "final_val_acc": final_validation_accuracy,
            "init_sha256": init_sha256,
            "state_sha256": state_dict_sha256(final_state),
            "config": config,
        },
        output_dir / "final.pt",
    )
    summary = {
        "total_steps": global_step,
        "best_val_loss": best_validation_loss,
        "best_val_acc": best_validation_accuracy,
        "final_val_loss": final_validation_loss,
        "final_val_acc": final_validation_accuracy,
        "fitting_samples": len(fitting_dataset),
        "validation_samples": len(validation_dataset),
        "fitting_positive": positive_fitting,
        "validation_positive": positive_validation,
        "trainable_params": model.trainable_params(),
        "init_sha256": init_sha256,
        "final_state_sha256": state_dict_sha256(final_state),
        "wandb_run_id": args.wandb_run_id,
        "wall_time_seconds": time.time() - training_started,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    LOGGER.info("Summary: %s", json.dumps(summary, sort_keys=True))
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
