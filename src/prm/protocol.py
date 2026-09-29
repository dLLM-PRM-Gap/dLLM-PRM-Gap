"""Reproducibility helpers shared by rebuttal experiments.

The original sharded evaluators seeded once with ``base_seed + shard_id``.
That makes a problem's samples depend on the number of shards and traversal
order.  This module instead derives every stochastic stream from stable,
semantic identifiers and records enough protocol metadata to audit a run.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROTOCOL_SCHEMA_VERSION = 1
STABLE_SEED_SPEC = "sha256-first8(canonical-json-tuple)-signed63-v1"


def _canonical_seed_part(value: Any) -> Any:
    """Convert a seed component to a deterministic JSON value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, (list, tuple)):
        return [_canonical_seed_part(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_seed_part(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    return str(value)


def derive_seed(
    base_seed: int,
    global_problem_id: int | str,
    particle_id: int | str,
    stage: int | str,
    *extra: Any,
) -> int:
    """Derive a stable 63-bit PyTorch seed from semantic identifiers.

    Python's builtin ``hash`` is intentionally randomized between processes,
    so it must not be used for experiment seeds.  The locked rebuttal protocol
    uses the first eight bytes of SHA-256 over a canonical JSON tuple.
    """
    payload = (
        int(base_seed),
        _canonical_seed_part(global_problem_id),
        _canonical_seed_part(particle_id),
        _canonical_seed_part(stage),
        *(_canonical_seed_part(item) for item in extra),
    )
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()[:8]
    # torch.Generator.manual_seed accepts signed int64 values on every backend.
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def derive_row_seeds(
    base_seed: int,
    global_problem_id: int | str,
    particle_ids: Sequence[int | str],
    stage: int | str,
    *extra: Any,
) -> list[int]:
    """Vector convenience wrapper around :func:`derive_seed`."""
    return [
        derive_seed(base_seed, global_problem_id, particle_id, stage, *extra)
        for particle_id in particle_ids
    ]


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 << 20) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(
    path: str | os.PathLike[str], chunk_size: int = 8 << 20
) -> str:
    """Hash a directory as sorted ``(relative path, file SHA-256)`` pairs."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and not candidate.is_symlink()
    )
    if not files:
        raise ValueError(f"cannot hash empty directory {root}")
    digest = hashlib.sha256()
    for candidate in files:
        relative = candidate.relative_to(root).as_posix()
        file_digest = sha256_file(candidate, chunk_size=chunk_size)
        record = json.dumps(
            (relative, file_digest),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(record).to_bytes(8, "big"))
        digest.update(record)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def callable_sha256(fn: Callable[..., Any]) -> str:
    """Hash a prompt/extraction implementation for a run manifest."""
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        source = repr(fn)
    return sha256_text(source)


def _git_metadata(repo_root: str | os.PathLike[str]) -> dict[str, Any]:
    root = str(Path(repo_root).resolve())

    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", root, *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

    try:
        commit = run("rev-parse", "HEAD")
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(run("status", "--porcelain"))
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def _dependency_versions(names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def artifact_descriptor(
    path: str | os.PathLike[str],
    *,
    sha256: str | None = None,
    require_hash: bool = True,
    verify_supplied_hash: bool = False,
) -> dict[str, Any]:
    """Describe a checkpoint/model artifact for a protocol manifest.

    Directories are represented by their resolved path.  Their model shard
    hashes should be supplied by deployment tooling via ``sha256``; silently
    hashing an arbitrary directory would be underspecified.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    result: dict[str, Any] = {
        "path": str(resolved),
        "kind": "directory" if resolved.is_dir() else "file",
    }
    if resolved.is_file():
        result["size_bytes"] = resolved.stat().st_size
        computed = (
            sha256_file(resolved)
            if (require_hash and (sha256 is None or verify_supplied_hash))
            else None
        )
        if sha256 is not None and computed is not None and computed != sha256:
            raise ValueError(
                f"SHA-256 mismatch for {resolved}: expected {sha256}, got {computed}"
            )
        result["sha256"] = sha256 or computed
        result["sha256_scheme"] = "file-bytes"
    else:
        if require_hash and not sha256:
            raise ValueError(
                f"An explicit model hash is required for directory artifact {resolved}"
            )
        computed = (
            sha256_directory(resolved)
            if require_hash and verify_supplied_hash
            else None
        )
        if sha256 is not None and computed is not None and computed != sha256:
            raise ValueError(
                f"SHA-256 mismatch for {resolved}: expected {sha256}, got {computed}"
            )
        result["sha256"] = sha256 or computed
        result["sha256_scheme"] = "sorted-relative-path-and-file-sha256-v1"
    return result


def validate_artifact_attestation(
    attestation_path: str | os.PathLike[str],
    *,
    artifact_path: str | os.PathLike[str],
    expected_sha256: str,
) -> dict[str, Any]:
    """Validate a precomputed artifact hash without re-reading a large model.

    Formal multi-GPU runs use one immutable attestation produced before
    launch.  This avoids every shard simultaneously hashing the same 15 GB
    shared model directory while still retaining a content-addressed gate in
    every run manifest.
    """
    path = Path(attestation_path).expanduser().resolve()
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported artifact attestation: {path}")
    artifact = payload.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError(f"Attestation has no artifact descriptor: {path}")
    resolved_artifact = str(Path(artifact_path).expanduser().resolve())
    errors = []
    if artifact.get("path") != resolved_artifact:
        errors.append(
            f"path {artifact.get('path')!r} != {resolved_artifact!r}"
        )
    if artifact.get("sha256") != expected_sha256:
        errors.append("SHA-256 does not match the supplied formal lock")
    if artifact.get("kind") not in {"file", "directory"}:
        errors.append(f"unsupported kind {artifact.get('kind')!r}")
    if not artifact.get("sha256_scheme"):
        errors.append("missing sha256_scheme")
    if errors:
        raise ValueError(
            f"Artifact attestation mismatch for {path}: " + "; ".join(errors)
        )
    return {
        "attestation": artifact_descriptor(path),
        "artifact": dict(artifact),
        "computed_at_utc": payload.get("computed_at_utc"),
        "hostname": payload.get("hostname"),
    }


def build_protocol_manifest(
    *,
    repo_root: str | os.PathLike[str],
    experiment: str,
    config: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    prompt_template: str | None = None,
    extraction_fn: Callable[..., Any] | None = None,
    dependency_names: Sequence[str] = (
        "torch",
        "transformers",
        "datasets",
        "peft",
        "safetensors",
        "numpy",
    ),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable, auditable run manifest."""
    manifest: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "experiment": experiment,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_metadata(repo_root),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": _dependency_versions(dependency_names),
        },
        "stable_seed_spec": STABLE_SEED_SPEC,
        "config": _canonical_seed_part(dict(config)),
        "artifacts": dict(artifacts or {}),
        "prompt_template_sha256": (
            sha256_text(prompt_template) if prompt_template is not None else None
        ),
        "extraction_callable_sha256": (
            callable_sha256(extraction_fn) if extraction_fn is not None else None
        ),
    }
    if extra:
        manifest["extra"] = _canonical_seed_part(dict(extra))
    return manifest


def atomic_write_json(
    path: str | os.PathLike[str],
    payload: Mapping[str, Any] | Sequence[Any],
    *,
    indent: int = 2,
) -> None:
    """Atomically replace a JSON result so interrupted jobs remain resumable."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
