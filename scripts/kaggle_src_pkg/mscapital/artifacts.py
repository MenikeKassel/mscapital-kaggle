"""Reproducible experiment artifacts and manifest helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_hash(value: object) -> str:
    arr = np.asarray(value)
    payload = f"{arr.dtype.str}|{arr.shape}|".encode("ascii") + np.ascontiguousarray(arr).tobytes()
    return sha256_bytes(payload)


def feature_hash(names: list[str] | tuple[str, ...]) -> str:
    return sha256_bytes("\n".join(names).encode("utf-8"))


def file_fingerprint(path: str | Path, *, sample_bytes: int = 1 << 20) -> str:
    """Hash file metadata plus head/tail samples without reading huge datasets fully."""

    target = Path(path)
    stat = target.stat()
    digest = hashlib.sha256()
    digest.update(str(target.resolve()).encode("utf-8"))
    digest.update(str(stat.st_size).encode("ascii"))
    digest.update(str(stat.st_mtime_ns).encode("ascii"))
    with target.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if stat.st_size > sample_bytes:
            handle.seek(max(0, stat.st_size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def path_fingerprints(paths: list[str | Path] | tuple[str | Path, ...]) -> dict[str, str]:
    return {str(Path(path)): file_fingerprint(path) for path in paths}


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclass
class ExperimentManifest:
    experiment_id: str
    status: str = "created"
    protocol: str = "protocol-v2"
    git_sha: str = field(default_factory=git_sha)
    config_hash: str | None = None
    data_fingerprints: dict[str, str] = field(default_factory=dict)
    feature_hash: str | None = None
    train_months: tuple[int, int] | None = None
    valid_months: tuple[int, int] | None = None
    metric: str = "cosine_uncentered"
    scores: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    best_step: int | None = None
    best_progress: float | None = None
    runtime_seconds: float | None = None

    def write(self, directory: str | Path) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "manifest.json"
        target.write_text(json.dumps(asdict(self), indent=2, sort_keys=True, default=str), encoding="utf-8")
        return target


def save_predictions(
    path: str | Path,
    *,
    sample_id: object,
    pred: object,
    target: object | None = None,
    month: object | None = None,
    split: object | None = None,
) -> Path:
    payload: dict[str, object] = {"sample_id": np.asarray(sample_id), "pred": np.asarray(pred)}
    if target is not None:
        payload["target"] = np.asarray(target)
    if month is not None:
        payload["month"] = np.asarray(month)
    if split is not None:
        payload["split"] = np.asarray(split)
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target_path, **payload)
    return target_path
