"""Audited checkpoint loading for DiffusionPRM evaluators.

Supports both the original full ``best.pt`` checkpoints and the compact
``adapter.safetensors`` release format.  Unlike the historical loaders, this
module propagates ``pool_strategy`` and fails if supplied weights do not match
the instantiated model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .model import DiffusionPRM, OutcomeRewardModel
from .protocol import sha256_file, sha256_text


@dataclass(frozen=True)
class CheckpointLoadReport:
    checkpoint_path: str
    checkpoint_format: str
    config: dict[str, Any]
    supplied_keys: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    pool_strategy: str
    config_source: str
    config_path: str | None
    config_sha256: str
    config_artifact_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ORMCheckpointLoadReport:
    checkpoint_path: str
    checkpoint_format: str
    config: dict[str, Any]
    supplied_keys: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    config_source: str
    config_path: str | None
    config_sha256: str
    config_artifact_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointReadResult:
    state_dict: dict[str, torch.Tensor]
    config: dict[str, Any]
    checkpoint_format: str
    checkpoint_path: Path
    config_source: str
    config_path: Path | None
    config_sha256: str
    config_artifact_sha256: str | None


def _resolve_checkpoint(path: str | Path) -> tuple[Path, Path | None]:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        for filename in ("adapter.safetensors", "model.safetensors", "best.pt"):
            candidate = resolved / filename
            if candidate.is_file():
                return candidate, resolved / "config.json"
        raise FileNotFoundError(
            f"No adapter.safetensors, model.safetensors, or best.pt in {resolved}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    config_path = resolved.parent / "config.json"
    return resolved, config_path if config_path.is_file() else None


def _load_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None or not config_path.is_file():
        return {}
    with open(config_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, Mapping) and isinstance(raw.get("training_config"), Mapping):
        return dict(raw["training_config"])
    return dict(raw) if isinstance(raw, Mapping) else {}


def _config_content_sha256(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256_text(encoded)


def read_prm_checkpoint_with_provenance(path: str | Path) -> CheckpointReadResult:
    """Read weights/config and retain the exact source of checkpoint config."""
    checkpoint_path, config_path = _resolve_checkpoint(path)
    suffix = checkpoint_path.suffix.lower()
    config_source = "none"
    resolved_config_path: Path | None = None
    config_artifact_sha256: str | None = None
    if suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint_path), device="cpu")
        config = _load_config(config_path)
        if config_path is not None and config_path.is_file():
            config_source = "adjacent"
            resolved_config_path = config_path.resolve()
            config_artifact_sha256 = sha256_file(resolved_config_path)
        checkpoint_format = "safetensors_adapter"
    elif suffix == ".pt":
        raw = torch.load(
            checkpoint_path,
            weights_only=False,
            map_location="cpu",
            mmap=True,
        )
        if not isinstance(raw, Mapping):
            raise TypeError(f"Expected mapping checkpoint, got {type(raw)!r}")
        embedded_config = raw.get("config", {})
        config = dict(embedded_config) if isinstance(embedded_config, Mapping) else {}
        if config:
            config_source = "embedded"
            resolved_config_path = checkpoint_path
        else:
            config = _load_config(config_path)
            if config_path is not None and config_path.is_file():
                config_source = "adjacent"
                resolved_config_path = config_path.resolve()
                config_artifact_sha256 = sha256_file(resolved_config_path)
        state_candidate = raw.get(
            "model_state_dict",
            raw.get("trainable_state_dict", raw.get("model", raw)),
        )
        if not isinstance(state_candidate, Mapping):
            raise TypeError("Checkpoint model state is not a mapping")
        state_dict = dict(state_candidate)
        checkpoint_format = (
            "pt_full"
            if "model_state_dict" in raw
            else ("pt_trainable" if "trainable_state_dict" in raw else "pt_state_dict")
        )
    else:
        raise ValueError(f"Unsupported PRM checkpoint suffix: {checkpoint_path}")

    if not state_dict:
        raise ValueError(f"Checkpoint contains no model tensors: {checkpoint_path}")
    non_tensors = [
        key for key, value in state_dict.items() if not torch.is_tensor(value)
    ]
    if non_tensors:
        raise TypeError(
            f"Checkpoint state has non-tensor values (first 5): {non_tensors[:5]}"
        )
    return CheckpointReadResult(
        state_dict=dict(state_dict),
        config=config,
        checkpoint_format=checkpoint_format,
        checkpoint_path=checkpoint_path,
        config_source=config_source,
        config_path=resolved_config_path,
        config_sha256=_config_content_sha256(config),
        config_artifact_sha256=config_artifact_sha256,
    )


def read_prm_checkpoint(
    path: str | Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], str, Path]:
    """Read weights/config without constructing the Dream backbone."""
    result = read_prm_checkpoint_with_provenance(path)
    return (
        result.state_dict,
        result.config,
        result.checkpoint_format,
        result.checkpoint_path,
    )


def read_orm_checkpoint(
    path: str | Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], str, Path]:
    """Read an ORM checkpoint with the same explicit wrapper-key policy."""
    return read_prm_checkpoint(path)


def read_orm_checkpoint_with_provenance(path: str | Path) -> CheckpointReadResult:
    """Read an ORM checkpoint and retain adjacent-config provenance."""
    return read_prm_checkpoint_with_provenance(path)


def construct_diffusion_prm(backbone, config: Mapping[str, Any]) -> DiffusionPRM:
    """Instantiate the exact architecture encoded by a training config."""
    return DiffusionPRM(
        backbone,
        hidden_size=backbone.config.hidden_size,
        step_embed_dim=int(config.get("step_embed_dim", 256)),
        reward_hidden=int(config.get("reward_hidden", 1024)),
        lora_r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        causal=bool(config.get("causal", False)),
        no_step_embed=bool(config.get("no_step_embed", False)),
        no_mask_aware=bool(config.get("no_mask_aware", False)),
        pool_strategy=str(config.get("pool_strategy", "mean")),
    )


def construct_outcome_reward_model(
    backbone,
    config: Mapping[str, Any],
) -> OutcomeRewardModel:
    return OutcomeRewardModel(
        backbone,
        hidden_size=backbone.config.hidden_size,
        lora_r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
    )


def load_state_dict_audited(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    adapter_only: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Load weights and reject the silent zero-match failure mode."""
    model_keys = set(model.state_dict())
    supplied_keys = set(state_dict)
    matched = model_keys & supplied_keys
    if not matched:
        raise RuntimeError("Checkpoint did not match any model parameter")
    unexpected_before_load = sorted(supplied_keys - model_keys)
    if unexpected_before_load:
        raise RuntimeError(
            "Checkpoint has unexpected keys (first 10): "
            + ", ".join(unexpected_before_load[:10])
        )

    incompatible = model.load_state_dict(dict(state_dict), strict=False)
    unexpected = tuple(incompatible.unexpected_keys)
    missing = tuple(incompatible.missing_keys)
    if unexpected:
        raise RuntimeError(
            "Unexpected checkpoint keys after load (first 10): "
            + ", ".join(unexpected[:10])
        )
    if not adapter_only and missing:
        raise RuntimeError(
            "Full checkpoint is missing model keys (first 10): "
            + ", ".join(missing[:10])
        )
    if adapter_only:
        trainable_model_keys = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        absent_trainable = sorted(trainable_model_keys - supplied_keys)
        if absent_trainable:
            raise RuntimeError(
                "Adapter checkpoint is missing trainable keys (first 10): "
                + ", ".join(absent_trainable[:10])
            )
    return missing, unexpected


def load_diffusion_prm(
    *,
    checkpoint: str | Path,
    model_path: str,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool = True,
    attn_implementation: str = "sdpa",
) -> tuple[DiffusionPRM, CheckpointLoadReport]:
    """Construct and load a DiffusionPRM with an audited key match."""
    from dream.modeling_dream import DreamModel

    checkpoint_result = read_prm_checkpoint_with_provenance(checkpoint)
    state_dict = checkpoint_result.state_dict
    config = checkpoint_result.config
    checkpoint_format = checkpoint_result.checkpoint_format
    resolved = checkpoint_result.checkpoint_path
    target_device = torch.device(device)
    device_map: str | None
    if target_device.type == "cuda":
        device_map = "cuda"
    else:
        device_map = None
    backbone = DreamModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        torch_dtype=dtype,
        device_map=device_map,
        local_files_only=local_files_only,
    )
    model = construct_diffusion_prm(backbone, config)
    adapter_only = checkpoint_format in {"safetensors_adapter", "pt_trainable"}
    missing, unexpected = load_state_dict_audited(
        model, state_dict, adapter_only=adapter_only
    )

    # Reward/step heads are created on CPU even when the backbone uses a device
    # map.  Moving only these small modules avoids an unnecessary second move
    # of the 7B backbone.
    model.reward_head.to(target_device)
    if not model.no_step_embed:
        model.step_embed.to(target_device)
        model.step_proj.to(target_device)
    model.eval()
    report = CheckpointLoadReport(
        checkpoint_path=str(resolved),
        checkpoint_format=checkpoint_format,
        config=dict(config),
        supplied_keys=len(state_dict),
        missing_keys=missing,
        unexpected_keys=unexpected,
        pool_strategy=model.pool_strategy,
        config_source=checkpoint_result.config_source,
        config_path=(
            str(checkpoint_result.config_path)
            if checkpoint_result.config_path is not None
            else None
        ),
        config_sha256=checkpoint_result.config_sha256,
        config_artifact_sha256=checkpoint_result.config_artifact_sha256,
    )
    return model, report


def load_outcome_reward_model(
    *,
    checkpoint: str | Path,
    model_path: str,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool = True,
    attn_implementation: str = "sdpa",
) -> tuple[OutcomeRewardModel, ORMCheckpointLoadReport]:
    """Construct and load an ORM while rejecting silent zero-match loads."""
    from dream.modeling_dream import DreamModel

    checkpoint_result = read_orm_checkpoint_with_provenance(checkpoint)
    state_dict = checkpoint_result.state_dict
    config = checkpoint_result.config
    checkpoint_format = checkpoint_result.checkpoint_format
    resolved = checkpoint_result.checkpoint_path
    target_device = torch.device(device)
    backbone = DreamModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    model = construct_outcome_reward_model(backbone, config)
    adapter_only = checkpoint_format in {
        "safetensors_adapter",
        "pt_trainable",
    }
    missing, unexpected = load_state_dict_audited(
        model,
        state_dict,
        adapter_only=adapter_only,
    )
    # Match training and formal full-to-adapter equivalence exactly: wrap PEFT
    # while the backbone is on CPU, then perform one model-wide device move.
    model.to(target_device)
    model.eval()
    report = ORMCheckpointLoadReport(
        checkpoint_path=str(resolved),
        checkpoint_format=checkpoint_format,
        config=dict(config),
        supplied_keys=len(state_dict),
        missing_keys=missing,
        unexpected_keys=unexpected,
        config_source=checkpoint_result.config_source,
        config_path=(
            str(checkpoint_result.config_path)
            if checkpoint_result.config_path is not None
            else None
        ),
        config_sha256=checkpoint_result.config_sha256,
        config_artifact_sha256=checkpoint_result.config_artifact_sha256,
    )
    return model, report
