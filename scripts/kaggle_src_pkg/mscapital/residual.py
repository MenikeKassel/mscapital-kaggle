"""Canonical rolling OOF and outer-fold residual construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .artifacts import ExperimentManifest, array_hash
from .clean_baseline import (
    PRODUCTION_TABLE_WEIGHT,
    REALMLP_CONFIG_HASH,
    TABLE_CONFIG_HASH,
    apply_production_rule,
)
from .metrics import cosine_uncentered
from .metrics import rms_scale
from .splits import CANONICAL_ROLLING_SPLITS, ROLLING_WINDOWS, visible_oof_end


ROLLING_SOURCE_END = {
    name: split.refit_train.end for name, split in CANONICAL_ROLLING_SPLITS.items()
}
ROLLING_EXPERIMENT_IDS = {
    "R21_30": ("canonical-realmlp-r21_30", "canonical-table-r21_30"),
    "R31_40": ("canonical-realmlp-r31_40", "canonical-table-r31_40"),
    "R41_50": ("canonical-realmlp-r41_50", "canonical-table-r41_50"),
    "R51_60": ("c2-realmlp-epochs30-t3", "clean-table-v2-t3"),
    "R61_70": ("c4-scale-realmlp-r61_70", "c4-scale-table-r61_70"),
}
CANONICAL_BLOCK_ROWS = {
    "R21_30": 177647,
    "R31_40": 177098,
    "R41_50": 177945,
    "R51_60": 177542,
    "R61_70": 175704,
}
CANONICAL_TOTAL_ROWS = sum(CANONICAL_BLOCK_ROWS.values())


@dataclass(frozen=True)
class OOFBlock:
    name: str
    sample_id: np.ndarray
    month: np.ndarray
    target: np.ndarray
    baseline_oof: np.ndarray
    source_train_end: int

    def validate(self) -> None:
        arrays = [self.sample_id, self.month, self.target, self.baseline_oof]
        if len({np.asarray(a).reshape(-1).shape for a in arrays}) != 1:
            raise ValueError(f"{self.name}: OOF arrays must have equal length")
        months = np.asarray(self.month).reshape(-1)
        if months.size == 0 or months.min() <= self.source_train_end:
            raise ValueError(f"{self.name}: OOF contains a month not after source_train_end")
        sample_id = np.asarray(self.sample_id).reshape(-1)
        if np.unique(sample_id).size != sample_id.size:
            raise ValueError(f"{self.name}: OOF contains duplicate sample_id values")
        target = np.asarray(self.target, dtype=float).reshape(-1)
        pred = np.asarray(self.baseline_oof, dtype=float).reshape(-1)
        if not np.isfinite(target).all() or not np.isfinite(pred).all():
            raise ValueError(f"{self.name}: target and baseline_oof must be finite")
        # Accept descriptive names such as m21_30_train20 while checking the
        # exact registered month interval and source boundary.
        expected_by_end = {
            20: (21, 30), 30: (31, 40), 40: (41, 50),
            50: (51, 60), 60: (61, 70),
        }
        expected = expected_by_end.get(self.source_train_end)
        if expected is None or set(np.asarray(months, dtype=int).tolist()) != set(range(expected[0], expected[1] + 1)):
            raise ValueError(f"{self.name}: block does not match a registered rolling window")


@dataclass(frozen=True)
class CanonicalOOF:
    sample_id: np.ndarray
    month: np.ndarray
    target: np.ndarray
    baseline_oof: np.ndarray
    source_train_end: np.ndarray

    def validate(self) -> None:
        arrays = [self.sample_id, self.month, self.target, self.baseline_oof, self.source_train_end]
        if len({np.asarray(a).reshape(-1).shape for a in arrays}) != 1:
            raise ValueError("canonical OOF arrays must have equal length")
        sample_ids = np.asarray(self.sample_id).reshape(-1)
        if np.unique(sample_ids).size != sample_ids.size:
            raise ValueError("canonical OOF contains duplicate sample_id values")
        if not np.all(np.asarray(self.source_train_end) < np.asarray(self.month)):
            raise ValueError("canonical OOF contains future-leaking source_train_end")
        if not np.isfinite(np.asarray(self.target, dtype=float)).all() or not np.isfinite(np.asarray(self.baseline_oof, dtype=float)).all():
            raise ValueError("canonical OOF target and baseline_oof must be finite")

    def save(self, path: str | Path) -> Path:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            sample_id=np.asarray(self.sample_id),
            month=np.asarray(self.month),
            target=np.asarray(self.target),
            baseline_oof=np.asarray(self.baseline_oof),
            source_train_end=np.asarray(self.source_train_end),
        )
        return target


def build_canonical_oof(
    blocks: Iterable[OOFBlock], *, require_complete: bool = True
) -> CanonicalOOF:
    blocks = tuple(blocks)
    if not blocks:
        raise ValueError("at least one OOF block is required")
    for block in blocks:
        block.validate()
    source_ends = [block.source_train_end for block in blocks]
    if len(set(source_ends)) != len(source_ends):
        raise ValueError("canonical OOF must contain one block per source_train_end")
    if require_complete and set(source_ends) != set(ROLLING_SOURCE_END.values()):
        raise ValueError("canonical OOF requires all five registered rolling blocks")
    merged = CanonicalOOF(
        sample_id=np.concatenate([np.asarray(b.sample_id).reshape(-1) for b in blocks]),
        month=np.concatenate([np.asarray(b.month).reshape(-1) for b in blocks]),
        target=np.concatenate([np.asarray(b.target).reshape(-1) for b in blocks]),
        baseline_oof=np.concatenate([np.asarray(b.baseline_oof).reshape(-1) for b in blocks]),
        source_train_end=np.concatenate([
            np.full(np.asarray(b.sample_id).reshape(-1).shape, b.source_train_end, dtype=np.int16)
            for b in blocks
        ]),
    )
    order = np.lexsort((merged.sample_id, merged.month))
    canonical = CanonicalOOF(
        sample_id=merged.sample_id[order],
        month=merged.month[order],
        target=merged.target[order],
        baseline_oof=merged.baseline_oof[order],
        source_train_end=merged.source_train_end[order],
    )
    canonical.validate()
    if require_complete:
        if set(np.asarray(canonical.month, dtype=int)) != set(range(21, 71)):
            raise ValueError("canonical OOF must cover every month from 21 through 70")
        expected_source = {
            month: source_end
            for source_end, (start, end) in {
                20: (21, 30), 30: (31, 40), 40: (41, 50),
                50: (51, 60), 60: (61, 70),
            }.items()
            for month in range(start, end + 1)
        }
        for month, source_end in expected_source.items():
            observed = np.unique(canonical.source_train_end[canonical.month == month])
            if not np.array_equal(observed, np.array([source_end], dtype=observed.dtype)):
                raise ValueError(f"month {month} has the wrong rolling source_train_end")
    return canonical


def _load_component_artifact(
    directory: str | Path,
    *,
    split_name: str,
    component: str,
    allow_smoke_config: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    directory = Path(directory)
    split = CANONICAL_ROLLING_SPLITS[split_name]
    component_index = 0 if component == "realmlp" else 1
    expected_experiment = ROLLING_EXPERIMENT_IDS[split_name][component_index]
    expected_config_hash = REALMLP_CONFIG_HASH if component == "realmlp" else TABLE_CONFIG_HASH
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{component} {split_name} manifest.json is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "experiment_id": expected_experiment,
        "status": "complete",
        "train_months": list(split.refit_train.as_tuple()),
        "valid_months": list(split.outer_valid.as_tuple()),
    }
    if not allow_smoke_config:
        expected["config_hash"] = expected_config_hash
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{component} {split_name} manifest {key} does not match registry")
    if not manifest.get("data_fingerprints"):
        raise ValueError(f"{component} {split_name} manifest requires data_fingerprints")

    artifacts: dict[str, dict[str, np.ndarray]] = {}
    for role, filename, month_range in (
        ("inner", "inner_predictions.npz", split.inner_tune),
        ("outer", "predictions.npz", split.outer_valid),
    ):
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(f"{component} {split_name} {filename} is required")
        with np.load(path) as source:
            required = {"sample_id", "month", "target", "pred"}
            if not required.issubset(source.files):
                raise ValueError(f"{component} {split_name} {role} artifact is incomplete")
            artifact = {key: np.asarray(source[key]) for key in required}
        if set(np.unique(artifact["month"])) != set(
            range(month_range.start, month_range.end + 1)
        ):
            raise ValueError(f"{component} {split_name} {role} months do not match registry")
        if np.unique(artifact["sample_id"]).size != artifact["sample_id"].size:
            raise ValueError(f"{component} {split_name} {role} sample_id must be unique")
        if not np.isfinite(artifact["target"]).all() or not np.isfinite(artifact["pred"]).all():
            raise ValueError(f"{component} {split_name} {role} arrays must be finite")
        artifacts[role] = artifact
    return artifacts["inner"], artifacts["outer"], manifest


def _align_components(
    realmlp: dict[str, np.ndarray], table: dict[str, np.ndarray], label: str
) -> None:
    for key in ("sample_id", "month", "target"):
        if not np.array_equal(realmlp[key], table[key]):
            raise ValueError(f"{label}: RealMLP and Table {key} arrays must align exactly")


def build_clean_baseline_oof_block(
    realmlp_dir: str | Path,
    table_dir: str | Path,
    split_name: str,
    output_dir: str | Path,
    *,
    allow_smoke_config: bool = False,
) -> dict[str, Any]:
    """Build one strictly historical Clean Baseline rolling OOF block."""

    if split_name not in CANONICAL_ROLLING_SPLITS:
        raise KeyError(f"unknown canonical rolling split: {split_name}")
    inner_realmlp, outer_realmlp, realmlp_manifest = _load_component_artifact(
        realmlp_dir, split_name=split_name, component="realmlp",
        allow_smoke_config=allow_smoke_config,
    )
    inner_table, outer_table, table_manifest = _load_component_artifact(
        table_dir, split_name=split_name, component="table",
        allow_smoke_config=allow_smoke_config,
    )
    _align_components(inner_realmlp, inner_table, f"{split_name} inner")
    _align_components(outer_realmlp, outer_table, f"{split_name} outer")
    scale_realmlp = rms_scale(inner_realmlp["pred"])
    scale_table = rms_scale(inner_table["pred"])
    baseline_oof = apply_production_rule(
        outer_realmlp["pred"], outer_table["pred"],
        scale_realmlp=scale_realmlp, scale_table=scale_table,
        table_weight=PRODUCTION_TABLE_WEIGHT,
    )
    source_train_end = ROLLING_SOURCE_END[split_name]
    block = OOFBlock(
        split_name,
        outer_realmlp["sample_id"], outer_realmlp["month"], outer_realmlp["target"],
        baseline_oof, source_train_end,
    )
    block.validate()
    output = Path(output_dir) / split_name
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.npz"
    np.savez_compressed(
        prediction_path,
        sample_id=block.sample_id,
        month=block.month,
        target=block.target,
        baseline_oof=block.baseline_oof,
        source_train_end=np.full(block.month.shape, source_train_end, dtype=np.int16),
    )
    score = cosine_uncentered(block.baseline_oof, block.target)
    diagnostics = {
        "artifact_role": "smoke" if allow_smoke_config else "canonical",
        "split": split_name,
        "source_train_end": source_train_end,
        "rows": int(block.month.size),
        "scale_realmlp": scale_realmlp,
        "scale_table": scale_table,
        "cosine_uncentered": score,
        "prediction_hash": array_hash(block.baseline_oof),
        "artifact_hashes": {
            "sample_id": array_hash(block.sample_id),
            "month": array_hash(block.month),
            "target": array_hash(block.target),
            "baseline_oof": array_hash(block.baseline_oof),
            "source_train_end": array_hash(
                np.full(block.month.shape, source_train_end, dtype=np.int16)
            ),
        },
        "component_hashes": {
            "realmlp_inner": array_hash(inner_realmlp["pred"]),
            "table_inner": array_hash(inner_table["pred"]),
            "realmlp_outer": array_hash(outer_realmlp["pred"]),
            "table_outer": array_hash(outer_table["pred"]),
        },
        "source_manifests": {
            "realmlp": {
                key: realmlp_manifest.get(key)
                for key in ("experiment_id", "git_sha", "config_hash", "feature_hash", "data_fingerprints")
            },
            "table": {
                key: table_manifest.get(key)
                for key in ("experiment_id", "git_sha", "config_hash", "feature_hash", "data_fingerprints")
            },
        },
    }
    payload = json.dumps(diagnostics, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result_hash = hashlib.sha256(payload).hexdigest()
    split = CANONICAL_ROLLING_SPLITS[split_name]
    ExperimentManifest(
        experiment_id=f"clean-baseline-oof-{split_name.lower()}",
        status="smoke" if allow_smoke_config else "complete",
        config_hash=result_hash,
        data_fingerprints={
            "realmlp": hashlib.sha256(
                json.dumps(
                    realmlp_manifest["data_fingerprints"], sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "table": hashlib.sha256(
                json.dumps(
                    table_manifest["data_fingerprints"], sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
        train_months=split.refit_train.as_tuple(),
        valid_months=split.outer_valid.as_tuple(),
        scores={"cosine_uncentered": score},
        diagnostics=diagnostics,
    ).write(output)
    (output / "report.md").write_text(
        "\n".join(
            [
                f"# Clean Baseline rolling OOF - {split_name}", "",
                f"- source train end: `{source_train_end}`",
                f"- rows: `{block.month.size}`",
                f"- cosine_uncentered: `{score:.9f}`",
                f"- RealMLP/Table inner RMS scales: `{scale_realmlp:.12g}` / `{scale_table:.12g}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return diagnostics | {"artifact": str(prediction_path), "result_hash": result_hash}


def load_clean_baseline_oof_block(
    location: str | Path, split_name: str
) -> OOFBlock:
    """Load one formal block and verify its manifest and every persisted array."""

    if split_name not in CANONICAL_ROLLING_SPLITS:
        raise KeyError(f"unknown canonical rolling split: {split_name}")
    location = Path(location)
    directory = location if location.is_dir() else location.parent
    prediction_path = directory / "predictions.npz" if location.is_dir() else location
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{split_name}: manifest.json is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = CANONICAL_ROLLING_SPLITS[split_name]
    expected = {
        "experiment_id": f"clean-baseline-oof-{split_name.lower()}",
        "status": "complete",
        "train_months": list(split.refit_train.as_tuple()),
        "valid_months": list(split.outer_valid.as_tuple()),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{split_name}: block manifest {key} does not match registry")
    diagnostics = manifest.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError(f"{split_name}: block manifest diagnostics are required")
    expected_result_hash = hashlib.sha256(
        json.dumps(diagnostics, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if manifest.get("config_hash") != expected_result_hash:
        raise ValueError(f"{split_name}: block manifest result hash is invalid")
    if diagnostics.get("artifact_role") != "canonical":
        raise ValueError(f"{split_name}: smoke blocks cannot enter canonical OOF")
    if diagnostics.get("split") != split_name:
        raise ValueError(f"{split_name}: diagnostics split does not match registry")
    if not manifest.get("data_fingerprints"):
        raise ValueError(f"{split_name}: block data fingerprints are required")
    expected_fingerprints = {
        component: hashlib.sha256(
            json.dumps(
                diagnostics["source_manifests"][component]["data_fingerprints"],
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for component in ("realmlp", "table")
    }
    if manifest["data_fingerprints"] != expected_fingerprints:
        raise ValueError(f"{split_name}: block data fingerprints do not match sources")
    if not prediction_path.exists():
        raise FileNotFoundError(f"{split_name}: predictions.npz is required")
    with np.load(prediction_path) as source:
        required = {
            "sample_id", "month", "target", "baseline_oof", "source_train_end"
        }
        if set(source.files) != required:
            raise ValueError(f"{split_name}: block prediction schema is invalid")
        arrays = {key: np.asarray(source[key]) for key in required}
    source_ends = np.unique(np.asarray(arrays["source_train_end"], dtype=np.int16))
    if source_ends.size != 1:
        raise ValueError(f"{split_name}: source_train_end must be constant")
    block = OOFBlock(
        name=split_name,
        sample_id=arrays["sample_id"],
        month=arrays["month"],
        target=arrays["target"],
        baseline_oof=arrays["baseline_oof"],
        source_train_end=int(source_ends[0]),
    )
    block.validate()
    if block.source_train_end != ROLLING_SOURCE_END[split_name]:
        raise ValueError(f"{split_name}: source_train_end does not match registry")
    if diagnostics.get("rows") != int(block.month.size):
        raise ValueError(f"{split_name}: manifest row count does not match artifact")
    artifact_hashes = diagnostics.get("artifact_hashes", {})
    for key, value in arrays.items():
        if artifact_hashes.get(key) != array_hash(value):
            raise ValueError(f"{split_name}: {key} hash does not match manifest")
    return block


def write_canonical_oof_artifact(
    block_locations: dict[str, str | Path], output_path: str | Path, *,
    strict_counts: bool = False,
) -> dict[str, Any]:
    """Verify and merge the five formal rolling blocks with a signed manifest."""

    if set(block_locations) != set(CANONICAL_ROLLING_SPLITS):
        raise ValueError("canonical OOF requires exactly the five registered block names")
    blocks = {
        name: load_clean_baseline_oof_block(block_locations[name], name)
        for name in CANONICAL_ROLLING_SPLITS
    }
    canonical = build_canonical_oof(blocks.values())
    if strict_counts:
        for name, block in blocks.items():
            expected_rows = CANONICAL_BLOCK_ROWS[name]
            if block.sample_id.size != expected_rows:
                raise ValueError(
                    f"{name}: expected {expected_rows} competition rows, "
                    f"got {block.sample_id.size}"
                )
        if canonical.sample_id.size != CANONICAL_TOTAL_ROWS:
            raise ValueError(
                f"canonical OOF expected {CANONICAL_TOTAL_ROWS} competition rows, "
                f"got {canonical.sample_id.size}"
            )
    output = Path(output_path)
    canonical.save(output)
    artifact_hashes = {
        key: array_hash(getattr(canonical, key))
        for key in ("sample_id", "month", "target", "baseline_oof", "source_train_end")
    }
    source_manifests: dict[str, dict[str, Any]] = {}
    data_fingerprints: dict[str, str] = {}
    for name, location in block_locations.items():
        location = Path(location)
        directory = location if location.is_dir() else location.parent
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        source_manifests[name] = {
            key: manifest.get(key)
            for key in (
                "experiment_id", "git_sha", "config_hash", "data_fingerprints",
                "train_months", "valid_months",
            )
        }
        data_fingerprints[name] = str(manifest["config_hash"])
    diagnostics = {
        "artifact_role": "canonical_residual_oof",
        "strict_counts": bool(strict_counts),
        "rows": int(canonical.sample_id.size),
        "months": [21, 70],
        "blocks": {
            name: {
                "rows": int(block.month.size),
                "months": list(CANONICAL_ROLLING_SPLITS[name].outer_valid.as_tuple()),
                "source_train_end": block.source_train_end,
            }
            for name, block in blocks.items()
        },
        "artifact_hashes": artifact_hashes,
        "source_manifests": source_manifests,
    }
    result_hash = hashlib.sha256(
        json.dumps(diagnostics, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    ExperimentManifest(
        experiment_id="canonical-clean-baseline-oof",
        status="complete",
        config_hash=result_hash,
        data_fingerprints=data_fingerprints,
        valid_months=(21, 70),
        diagnostics=diagnostics,
    ).write(output.parent)
    (output.parent / "report.md").write_text(
        "\n".join(
            [
                "# Canonical Clean Baseline rolling OOF", "",
                f"- rows: `{canonical.sample_id.size}`",
                "- months: `21-70`",
                "- blocks: `R21_30 / R31_40 / R41_50 / R51_60 / R61_70`",
                "- every row satisfies: `source_train_end < month`", "",
            ]
        ),
        encoding="utf-8",
    )
    return diagnostics | {"output": str(output), "result_hash": result_hash}


def load_canonical_oof_artifact(path: str | Path) -> CanonicalOOF:
    """Load a canonical OOF artifact only after replaying its manifest checks."""

    path = Path(path)
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("canonical OOF manifest.json is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "experiment_id": "canonical-clean-baseline-oof",
        "status": "complete",
        "valid_months": [21, 70],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"canonical OOF manifest {key} is invalid")
    diagnostics = manifest.get("diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics.get("artifact_role") != "canonical_residual_oof":
        raise ValueError("canonical OOF diagnostics are invalid")
    if diagnostics.get("strict_counts"):
        if diagnostics.get("rows") != CANONICAL_TOTAL_ROWS:
            raise ValueError("strict canonical OOF row count is invalid")
        for name, expected_rows in CANONICAL_BLOCK_ROWS.items():
            if diagnostics.get("blocks", {}).get(name, {}).get("rows") != expected_rows:
                raise ValueError(f"strict canonical OOF {name} row count is invalid")
    result_hash = hashlib.sha256(
        json.dumps(diagnostics, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if manifest.get("config_hash") != result_hash:
        raise ValueError("canonical OOF manifest result hash is invalid")
    if manifest.get("data_fingerprints") != {
        name: str(diagnostics["source_manifests"][name]["config_hash"])
        for name in CANONICAL_ROLLING_SPLITS
    }:
        raise ValueError("canonical OOF source fingerprints are incomplete")
    with np.load(path) as source:
        required = {"sample_id", "month", "target", "baseline_oof", "source_train_end"}
        if set(source.files) != required:
            raise ValueError("canonical OOF NPZ schema is invalid")
        arrays = {key: np.asarray(source[key]) for key in required}
    for key, value in arrays.items():
        if diagnostics.get("artifact_hashes", {}).get(key) != array_hash(value):
            raise ValueError(f"canonical OOF {key} hash does not match manifest")
    canonical = CanonicalOOF(**arrays)
    canonical.validate()
    if canonical.sample_id.size != diagnostics.get("rows"):
        raise ValueError("canonical OOF row count does not match manifest")
    if set(np.asarray(canonical.month, dtype=int)) != set(range(21, 71)):
        raise ValueError("canonical OOF does not cover months 21-70")
    return canonical


def rolling_window_spec() -> tuple[tuple[str, tuple[int, int], tuple[int, int]], ...]:
    return tuple((name, train.as_tuple(), valid.as_tuple()) for name, train, valid in ROLLING_WINDOWS)


def outer_residual(
    canonical: CanonicalOOF,
    outer_name: str,
) -> dict[str, np.ndarray | float]:
    """Return the historical residual view visible to one outer fold."""

    canonical.validate()
    end = visible_oof_end(outer_name)
    mask = np.asarray(canonical.month) <= end
    if not mask.any():
        raise ValueError(f"no canonical OOF rows visible to {outer_name}")
    p = np.asarray(canonical.baseline_oof, dtype=np.float64)[mask]
    y = np.asarray(canonical.target, dtype=np.float64)[mask]
    denominator = float(np.dot(p, p))
    beta = 0.0 if denominator == 0 else float(np.dot(p, y) / denominator)
    residual = y - beta * p
    score = cosine_uncentered(beta * p, y)
    return {
        "sample_id": np.asarray(canonical.sample_id)[mask],
        "month": np.asarray(canonical.month)[mask],
        "target": y,
        "baseline_oof": p,
        "residual": residual,
        "beta": beta,
        "baseline_cosine": score,
    }
