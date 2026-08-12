"""Configuration and path resolution for protocol-v2.

No experiment code is allowed to bake in a workstation-specific drive.  A
config file can override the environment variables, while the environment
variables provide a convenient local default.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _path(value: str | os.PathLike[str] | None, fallback: str) -> Path:
    return Path(value if value is not None else os.environ.get(fallback, ".")).expanduser()


@dataclass(frozen=True)
class ProjectConfig:
    data_root: Path = field(default_factory=lambda: _path(None, "MSCAP_DATA_ROOT"))
    reference_root: Path = field(
        default_factory=lambda: _path(None, "MSCAP_REFERENCE_ROOT")
    )
    artifact_root: Path = field(
        default_factory=lambda: _path("output/experiments", "MSCAP_ARTIFACT_ROOT")
    )
    protocol: str = "protocol-v2"
    target_round: int | None = 4
    quantile_bins: int = 20
    protected_columns: tuple[str, ...] = ("sample_id", "month", "target")

    def resolved(self) -> "ProjectConfig":
        return ProjectConfig(
            data_root=self.data_root.resolve(),
            reference_root=self.reference_root.resolve(),
            artifact_root=self.artifact_root.resolve(),
            protocol=self.protocol,
            target_round=self.target_round,
            quantile_bins=self.quantile_bins,
            protected_columns=tuple(self.protected_columns),
        )

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("data_root", "reference_root", "artifact_root"):
            value[key] = str(value[key])
        value["protected_columns"] = list(value["protected_columns"])
        return value


def _load_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without optional dep
        raise RuntimeError(
            "YAML config requires PyYAML; use JSON or install mscapital[yaml]."
        ) from exc
    return yaml.safe_load(text) or {}


def load_config(path: str | os.PathLike[str] | None = None) -> ProjectConfig:
    """Load a JSON/YAML config and apply environment-backed defaults."""

    if path is None:
        return ProjectConfig().resolved()
    mapping = dict(_load_mapping(Path(path)))
    defaults = ProjectConfig()
    protected = mapping.get("protected_columns", defaults.protected_columns)
    cfg = ProjectConfig(
        data_root=Path(mapping.get("data_root", defaults.data_root)),
        reference_root=Path(mapping.get("reference_root", defaults.reference_root)),
        artifact_root=Path(mapping.get("artifact_root", defaults.artifact_root)),
        protocol=str(mapping.get("protocol", defaults.protocol)),
        target_round=mapping.get("target_round", defaults.target_round),
        quantile_bins=int(mapping.get("quantile_bins", defaults.quantile_bins)),
        protected_columns=tuple(protected),
    )
    if cfg.quantile_bins < 2:
        raise ValueError("quantile_bins must be >= 2")
    return cfg.resolved()


def config_hash(config: ProjectConfig | Mapping[str, Any]) -> str:
    payload = config.as_dict() if isinstance(config, ProjectConfig) else dict(config)
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
