"""Contracts for the matched binary-versus-fractional continuation scorers.

The formal comparison changes one field only: the target used by soft BCE.
Everything else -- frozen states, architecture, initialization within a seed,
optimizer, update count, and checkpoint-selection rule -- is content locked.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from prm.protocol import sha256_text


SCHEMA_VERSION = 1
TRAINING_PROTOCOL_ID = "gsm8k_continuation_matched_soft_bce_v1"
PAIR_REPORT_SCHEMA_VERSION = 1
INIT_LOCK_SCHEMA_VERSION = 1
LOCKED_SPLIT_SHA256 = (
    "d2717f135eb263bd07983bd97713f22ce6d0051705e84b461b8761e9053b3ffd"
)
FORMAL_TRAINING_SEEDS = (42, 43)
ROLES = {
    "binary": "binary_label",
    "fractional": "fractional_label",
}
GROUP_COUNTS = {
    "fitting": {"states": 8000, "problems": 800},
    "validation": {"states": 2000, "problems": 200},
}
ARCHITECTURE_CONFIG = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "step_embed_dim": 256,
    "reward_hidden": 1024,
    "causal": False,
    "no_step_embed": False,
    "no_mask_aware": False,
    "pool_strategy": "mean",
}
FORMAL_TRAINING_CONFIG = {
    "epochs": 2,
    "batch_size": 4,
    "grad_accum": 8,
    "lr": 1e-5,
    "weight_decay": 0.01,
    "warmup_fraction": 0.0,
    "max_grad_norm": 1.0,
    "max_steps": 500,
    "max_val_samples": 0,
    "save_every_steps": 250,
    "eval_every_steps": 250,
    "log_every_steps": 25,
    **ARCHITECTURE_CONFIG,
}
CHECKPOINT_SELECTION_METRIC = "validation_soft_bce"
CONTINUATION_SAMPLER_CONFIG = {
    "steps": 128,
    "temperature": 0.5,
    "top_p": 1.0,
    "top_k": None,
    "alg": "maskgit_plus",
    "alg_temp": 0.5,
    "mask_token_id": 151666,
    "gen_length": 512,
    "eps": 1e-5,
}


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolved(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def formal_training_config(
    *, role: str, seed: int, init_sha256: str
) -> dict[str, Any]:
    if role not in ROLES:
        raise ValueError(f"unsupported matched scorer role: {role}")
    return {
        **FORMAL_TRAINING_CONFIG,
        "label_field": ROLES[role],
        "seed": int(seed),
        "model_init_seed": int(seed),
        "expected_init_sha256": init_sha256,
    }


def pair_identity(material: Mapping[str, Any]) -> str:
    """Return a stable identity for one seed's two-condition comparison."""
    return sha256_text(
        json.dumps(
            dict(material),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )


def validate_merge_manifest(
    manifest: Mapping[str, Any],
    *,
    group: str,
    state_path: str | Path,
    state_sha256: str,
) -> dict[str, Any]:
    """Validate a strict continuation merge used by formal training."""
    if group not in GROUP_COUNTS:
        raise ValueError(f"unsupported continuation group: {group}")
    expected = GROUP_COUNTS[group]
    errors: list[str] = []
    if manifest.get("schema_version") != 2:
        errors.append("merge schema is not version 2")
    if manifest.get("group") != group:
        errors.append(f"merge group is not {group}")
    if manifest.get("formal_counts_enforced") is not True:
        errors.append("merge did not enforce formal counts")
    split = manifest.get("split_manifest")
    if (
        not isinstance(split, Mapping)
        or split.get("sha256") != LOCKED_SPLIT_SHA256
    ):
        errors.append("merge uses the wrong locked continuation split")
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or not source_commit:
        errors.append("merge lacks source commit")
    regular = manifest.get("regular")
    if not isinstance(regular, Mapping):
        errors.append("merge lacks regular-state descriptor")
    else:
        if regular.get("path") != _resolved(state_path):
            errors.append("regular-state path mismatch")
        if regular.get("sha256") != state_sha256:
            errors.append("regular-state hash mismatch")
        if regular.get("states") != expected["states"]:
            errors.append("regular-state count mismatch")
        if regular.get("problems") != expected["problems"]:
            errors.append("regular problem count mismatch")
        if regular.get("training_eligible") is not True:
            errors.append("regular states are not marked training eligible")
        bucket_counts = regular.get("states_per_bucket")
        if not isinstance(bucket_counts, Mapping) or any(
            bucket_counts.get(str(bucket)) != expected["problems"]
            for bucket in range(10)
        ):
            errors.append("regular states are not balanced over buckets 0..9")
    terminal = manifest.get("terminal_exact_zero")
    if not isinstance(terminal, Mapping):
        errors.append("merge lacks exact-zero descriptor")
    elif (
        terminal.get("states") != expected["problems"]
        or terminal.get("problems") != expected["problems"]
        or terminal.get("training_eligible") is not False
    ):
        errors.append("exact-zero exclusion/count contract mismatch")
    if manifest.get("base_seed") != 42:
        errors.append("merge base seed is not the frozen value 42")
    if manifest.get("rollout_batch_size") != 8:
        errors.append("merge rollout batch size is not the frozen value 8")
    if manifest.get("sampler") != CONTINUATION_SAMPLER_CONFIG:
        errors.append("merge sampler differs from the frozen continuation protocol")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        errors.append("merge lacks generation artifacts")
    else:
        dream = artifacts.get("dream_backbone")
        dream_attestation = artifacts.get("dream_backbone_attestation")
        if not isinstance(dream, Mapping) or not _valid_sha256(
            dream.get("sha256")
        ):
            errors.append("merge lacks Dream backbone hash")
        if not isinstance(dream_attestation, Mapping) or not _valid_sha256(
            dream_attestation.get("sha256")
        ):
            errors.append("merge lacks Dream hash-attestation artifact")
    if errors:
        raise ValueError(
            f"formal {group} continuation merge failed: " + "; ".join(errors)
        )
    return {
        "group": group,
        "states": expected["states"],
        "problems": expected["problems"],
        "source_commit": source_commit,
        "split_sha256": LOCKED_SPLIT_SHA256,
        "sampler": dict(manifest["sampler"]),
        "base_seed": manifest.get("base_seed"),
        "rollout_batch_size": manifest.get("rollout_batch_size"),
        "generation_artifacts": dict(artifacts),
    }


def validate_state_records(
    records: Sequence[Mapping[str, Any]], *, group: str
) -> dict[str, Any]:
    """Recheck the immutable R=8 target and state cardinality at training time."""
    if group not in GROUP_COUNTS:
        raise ValueError(f"unsupported continuation group: {group}")
    expected = GROUP_COUNTS[group]
    if len(records) != expected["states"]:
        raise ValueError(
            f"{group} has {len(records)} states, expected {expected['states']}"
        )
    state_ids: set[str] = set()
    by_problem: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    bucket_counts: Counter[int] = Counter()
    for index, record in enumerate(records):
        state_id = str(record.get("state_id", ""))
        problem_id = str(record.get("problem_id", ""))
        if not state_id or state_id in state_ids:
            raise ValueError(f"duplicate/missing state_id at {group}[{index}]")
        if not problem_id:
            raise ValueError(f"missing problem_id at {group}[{index}]")
        state_ids.add(state_id)
        if record.get("group") != group:
            raise ValueError(f"{state_id} has wrong continuation group")
        required = (
            "input_ids",
            "prompt_len",
            "mask_ratio",
            "mask_bucket",
            "binary_label",
            "fractional_label",
            "continuation_outcomes",
        )
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"{state_id} lacks required fields: {missing}")
        input_ids = torch.as_tensor(record["input_ids"]).view(-1)
        prompt_len = int(record["prompt_len"])
        if prompt_len <= 0 or prompt_len >= len(input_ids):
            raise ValueError(f"{state_id} has invalid prompt_len")
        mask_ratio = float(record["mask_ratio"])
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"{state_id} is not a regular nonterminal state")
        bucket = int(record["mask_bucket"])
        if bucket not in range(10):
            raise ValueError(f"{state_id} has invalid mask bucket")
        if min(int(mask_ratio * 10), 9) != bucket:
            raise ValueError(f"{state_id} mask ratio/bucket mismatch")
        binary = float(record["binary_label"])
        if binary not in {0.0, 1.0}:
            raise ValueError(f"{state_id} has a non-binary inherited target")
        outcomes = list(record["continuation_outcomes"])
        if len(outcomes) != 8 or any(
            float(outcome) not in {0.0, 1.0} for outcome in outcomes
        ):
            raise ValueError(f"{state_id} does not have exactly R=8 outcomes")
        fractional = float(record["fractional_label"])
        if not math.isclose(
            fractional,
            sum(float(outcome) for outcome in outcomes) / 8,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{state_id} has an invalid fractional target")
        by_problem[problem_id].append(record)
        bucket_counts[bucket] += 1
    if len(by_problem) != expected["problems"]:
        raise ValueError(
            f"{group} has {len(by_problem)} problems, expected "
            f"{expected['problems']}"
        )
    for problem_id, rows in by_problem.items():
        if len(rows) != 10 or Counter(
            int(row["mask_bucket"]) for row in rows
        ) != Counter(range(10)):
            raise ValueError(
                f"{problem_id} does not have exactly one state per bucket"
            )
        if len({float(row["binary_label"]) for row in rows}) != 1:
            raise ValueError(f"{problem_id} inherited labels differ across states")
    if any(
        bucket_counts[bucket] != expected["problems"] for bucket in range(10)
    ):
        raise ValueError(f"{group} is not balanced over mask buckets")
    return {
        "states": len(records),
        "problems": len(by_problem),
        "problem_ids": sorted(by_problem),
        "state_ids_sha256": sha256_text(
            json.dumps(sorted(state_ids), separators=(",", ":"))
        ),
        "r": 8,
    }


def validate_init_lock(
    lock: Mapping[str, Any],
    *,
    seed: int,
    model_sha256: str,
    source_git_commit: str | None = None,
) -> str:
    errors: list[str] = []
    if lock.get("schema_version") != INIT_LOCK_SCHEMA_VERSION:
        errors.append("unsupported initialization-lock schema")
    if lock.get("protocol_id") != TRAINING_PROTOCOL_ID:
        errors.append("initialization-lock protocol mismatch")
    if lock.get("complete") is not True:
        errors.append("initialization lock is incomplete")
    if lock.get("seed") != int(seed) or lock.get("model_init_seed") != int(seed):
        errors.append("initialization-lock seed mismatch")
    if lock.get("model_sha256") != model_sha256:
        errors.append("initialization-lock Dream hash mismatch")
    if lock.get("architecture") != ARCHITECTURE_CONFIG:
        errors.append("initialization-lock architecture mismatch")
    init_sha256 = lock.get("init_sha256")
    if not _valid_sha256(init_sha256):
        errors.append("initialization lock has invalid trainable-state hash")
    git = lock.get("git")
    if (
        not isinstance(git, Mapping)
        or not git.get("commit")
        or git.get("dirty") is not False
    ):
        errors.append("initialization lock was not produced from clean source")
    elif source_git_commit is not None and git.get("commit") != source_git_commit:
        errors.append("initialization-lock source commit mismatch")
    if errors:
        raise ValueError("matched initialization gate failed: " + "; ".join(errors))
    return str(init_sha256)


def validate_pair_report(
    report: Mapping[str, Any],
    *,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    if report.get("schema_version") != PAIR_REPORT_SCHEMA_VERSION:
        errors.append("unsupported pair-report schema")
    if report.get("protocol_id") != TRAINING_PROTOCOL_ID:
        errors.append("pair-report protocol mismatch")
    if report.get("all_passed") is not True or report.get("complete") is not True:
        errors.append("pair report is not complete and passing")
    seed = report.get("seed")
    if seed not in FORMAL_TRAINING_SEEDS:
        errors.append("pair report uses a non-formal seed")
    if expected_seed is not None and seed != expected_seed:
        errors.append("pair-report seed mismatch")
    pair_id = report.get("pair_id")
    init_sha256 = report.get("init_sha256")
    if not _valid_sha256(pair_id) or not _valid_sha256(init_sha256):
        errors.append("pair report has invalid pair/init identity")
    if report.get("architecture") != ARCHITECTURE_CONFIG:
        errors.append("pair-report architecture mismatch")
    if report.get("locked_split_sha256") != LOCKED_SPLIT_SHA256:
        errors.append("pair-report locked split mismatch")
    if not report.get("source_git_commit") or not report.get(
        "continuation_source_commit"
    ):
        errors.append("pair report lacks source commits")
    git = report.get("git")
    if (
        not isinstance(git, Mapping)
        or git.get("dirty") is not False
        or git.get("commit") != report.get("source_git_commit")
    ):
        errors.append("pair report was not produced from its clean source commit")
    only_difference = report.get("only_intended_difference")
    if only_difference != {
        "field": "label_field",
        "binary": "binary_label",
        "fractional": "fractional_label",
    }:
        errors.append("pair report does not lock the sole target-field difference")
    common_artifacts = report.get("common_artifact_sha256")
    if not isinstance(common_artifacts, Mapping) or any(
        not _valid_sha256(common_artifacts.get(name))
        for name in (
            "dream_backbone",
            "dream_backbone_hash_attestation",
            "fitting_states",
            "validation_states",
            "fitting_merge_manifest",
            "validation_merge_manifest",
            "initialization_lock",
        )
    ):
        errors.append("pair report lacks common content-addressed artifacts")
    common_config = report.get("common_training_config")
    if not isinstance(common_config, Mapping) or dict(common_config) != (
        FORMAL_TRAINING_CONFIG
    ):
        errors.append("pair-report common training config mismatch")
    entries = report.get("scorers")
    if not isinstance(entries, Mapping) or set(entries) != set(ROLES):
        errors.append("pair report must contain binary and fractional scorers")
    else:
        for role, label_field in ROLES.items():
            entry = entries.get(role)
            if not isinstance(entry, Mapping):
                errors.append(f"pair report lacks {role} entry")
                continue
            if entry.get("role") != role or entry.get("label_field") != label_field:
                errors.append(f"pair-report {role} role/label mismatch")
            for key in (
                "checkpoint_sha256",
                "provenance_sha256",
                "checkpoint_config_provenance_sha256",
            ):
                if not _valid_sha256(entry.get(key)):
                    errors.append(f"pair-report {role} has invalid {key}")
            if entry.get("pair_id") != pair_id or entry.get(
                "init_sha256"
            ) != init_sha256:
                errors.append(f"pair-report {role} pair/init mismatch")
            if entry.get("seed") != seed:
                errors.append(f"pair-report {role} seed mismatch")
            if not entry.get("checkpoint_path") or not entry.get(
                "provenance_path"
            ):
                errors.append(f"pair-report {role} lacks artifact paths")
            if entry.get("checkpoint_step") not in {250, 500}:
                errors.append(f"pair-report {role} checkpoint step mismatch")
            soft_bce = entry.get("validation_soft_bce")
            if not isinstance(soft_bce, (int, float)) or not math.isfinite(
                float(soft_bce)
            ):
                errors.append(
                    f"pair-report {role} validation soft BCE is not finite"
                )
    if errors:
        raise ValueError("matched pair-report gate failed: " + "; ".join(errors))
    return {
        "all_passed": True,
        "seed": int(seed),
        "pair_id": str(pair_id),
        "init_sha256": str(init_sha256),
    }


def validate_matched_checkpoint_gate(
    *,
    role: str,
    checkpoint_sha256: str,
    checkpoint_report: Mapping[str, Any],
    provenance: Mapping[str, Any],
    provenance_path: str | Path,
    provenance_sha256: str,
    pair_report: Mapping[str, Any],
    pair_report_path: str | Path,
    pair_report_sha256: str,
) -> dict[str, Any]:
    """Fail closed unless scoring uses one member of an audited matched pair."""
    if role not in ROLES:
        raise ValueError(f"unsupported matched scorer role: {role}")
    errors: list[str] = []
    if provenance.get("schema_version") != SCHEMA_VERSION:
        errors.append("training provenance schema mismatch")
    if provenance.get("protocol_id") != TRAINING_PROTOCOL_ID:
        errors.append("training provenance protocol mismatch")
    if provenance.get("complete") is not True:
        errors.append("training provenance is incomplete")
    if provenance.get("role") != role or provenance.get("label_field") != ROLES[role]:
        errors.append("training provenance role/label mismatch")
    seed = provenance.get("seed")
    if seed not in FORMAL_TRAINING_SEEDS:
        errors.append("training provenance uses a non-formal seed")
    init_sha256 = provenance.get("init_sha256")
    pair_id = provenance.get("pair_id")
    if not _valid_sha256(init_sha256) or not _valid_sha256(pair_id):
        errors.append("training provenance has invalid pair/init identity")
    expected_training = (
        formal_training_config(
            role=role,
            seed=int(seed),
            init_sha256=str(init_sha256),
        )
        if seed in FORMAL_TRAINING_SEEDS and _valid_sha256(init_sha256)
        else None
    )
    if expected_training is None or provenance.get("training_config") != (
        expected_training
    ):
        errors.append("training provenance config is not the frozen formal config")
    checkpoint = provenance.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        errors.append("training provenance lacks checkpoint descriptor")
    else:
        if checkpoint.get("sha256") != checkpoint_sha256:
            errors.append("training provenance checkpoint hash mismatch")
        if checkpoint.get("path") != checkpoint_report.get("checkpoint_path"):
            errors.append("runtime checkpoint path differs from provenance")
        if checkpoint.get("selection_metric") != CHECKPOINT_SELECTION_METRIC:
            errors.append("checkpoint was not selected by validation soft BCE")
        if checkpoint.get("step") not in {250, 500}:
            errors.append("selected checkpoint step is not an audited eval boundary")
        soft_bce = checkpoint.get("validation_soft_bce")
        if not isinstance(soft_bce, (int, float)) or not math.isfinite(
            float(soft_bce)
        ):
            errors.append("selected validation soft BCE is not finite")
    if checkpoint_report.get("checkpoint_format") != "pt_trainable":
        errors.append("matched scorer must use the trainable-only best.pt")
    config = checkpoint_report.get("config")
    if not isinstance(config, Mapping):
        errors.append("checkpoint load report lacks training config")
    else:
        if expected_training is not None:
            for key, value in expected_training.items():
                if config.get(key) != value:
                    errors.append(f"checkpoint config {key} mismatch")
        if config.get("formal_provenance") != provenance.get(
            "checkpoint_config_provenance"
        ):
            errors.append("checkpoint-embedded formal provenance mismatch")
        if config.get("init_sha256") != init_sha256:
            errors.append("checkpoint config initialization hash mismatch")
        for key, value in ARCHITECTURE_CONFIG.items():
            if config.get(key) != value:
                errors.append(f"checkpoint architecture {key} mismatch")
    embedded = provenance.get("checkpoint_config_provenance")
    if not isinstance(embedded, Mapping):
        errors.append("training provenance lacks checkpoint config provenance")
        embedded_sha256 = None
    else:
        embedded_sha256 = sha256_text(
            json.dumps(
                dict(embedded),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )

    try:
        pair_gate = validate_pair_report(
            pair_report,
            expected_seed=int(seed) if isinstance(seed, int) else None,
        )
    except ValueError as exc:
        errors.append(str(exc))
        pair_gate = {}
    entry = (
        pair_report.get("scorers", {}).get(role)
        if isinstance(pair_report.get("scorers"), Mapping)
        else None
    )
    if not isinstance(entry, Mapping):
        errors.append(f"pair report lacks {role} scorer")
    else:
        if entry.get("checkpoint_sha256") != checkpoint_sha256:
            errors.append("pair-report checkpoint hash mismatch")
        if entry.get("provenance_sha256") != provenance_sha256:
            errors.append("pair-report provenance hash mismatch")
        if entry.get("provenance_path") != _resolved(provenance_path):
            errors.append("pair-report provenance path mismatch")
        if entry.get("checkpoint_config_provenance_sha256") != (
            embedded_sha256
        ):
            errors.append("pair-report embedded provenance hash mismatch")
        if entry.get("pair_id") != pair_id or entry.get(
            "init_sha256"
        ) != init_sha256:
            errors.append("pair-report scorer identity mismatch")
    if pair_gate and pair_gate.get("pair_id") != pair_id:
        errors.append("pair-report pair identity differs from provenance")
    if pair_report.get("locked_split_sha256") != provenance.get(
        "locked_split_sha256"
    ):
        errors.append("pair-report split identity differs from provenance")
    if pair_report.get("source_git_commit") != provenance.get(
        "source_git_commit"
    ):
        errors.append("pair-report source commit differs from provenance")
    if pair_report.get("continuation_source_commit") != provenance.get(
        "continuation_source_commit"
    ):
        errors.append("pair-report continuation commit differs from provenance")
    if pair_report.get("common_artifact_sha256") != provenance.get(
        "artifact_sha256"
    ):
        errors.append("pair-report artifacts differ from provenance")
    pair_artifact = pair_report.get("artifact")
    if isinstance(pair_artifact, Mapping):
        # A report must not claim a self hash: such a value cannot be stable
        # under serialization and is easy to misinterpret as an external lock.
        errors.append("pair report must not contain a self-referential artifact")
    if not _valid_sha256(pair_report_sha256):
        errors.append("pair-report artifact hash is invalid")
    if not _valid_sha256(provenance_sha256):
        errors.append("training-provenance artifact hash is invalid")
    if not Path(pair_report_path).expanduser().resolve().is_file():
        errors.append("pair-report artifact is missing")
    if errors:
        raise ValueError(
            f"matched-{role} checkpoint gate failed: " + "; ".join(errors)
        )
    return {
        "all_passed": True,
        "protocol_id": TRAINING_PROTOCOL_ID,
        "role": role,
        "seed": int(seed),
        "pair_id": str(pair_id),
        "init_sha256": str(init_sha256),
        "checkpoint_sha256": checkpoint_sha256,
        "provenance_sha256": provenance_sha256,
        "pair_report_sha256": pair_report_sha256,
        "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
    }
