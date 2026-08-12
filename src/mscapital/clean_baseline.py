"""C4 nested calibration and Clean Baseline v2 freezing."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import ExperimentManifest, array_hash, save_predictions
from .ensemble import EnsembleCalibrator
from .metrics import cosine_uncentered
from .splits import NESTED_SPLITS


METHODS = ("raw", "std", "rms")
WEIGHT_GRID = np.arange(0.0, 1.0001, 0.01)


def _load_aligned(
    realmlp_path: Path,
    table_path: Path,
    *,
    expected_months: tuple[int, int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    realmlp_npz = np.load(realmlp_path)
    table_npz = np.load(table_path)
    required = {"sample_id", "month", "target", "pred"}
    for name, artifact in (("RealMLP", realmlp_npz), ("Table", table_npz)):
        if not required.issubset(artifact.files):
            raise ValueError(f"{name} artifact is missing {sorted(required - set(artifact.files))}")
        for key in ("target", "pred"):
            if not np.isfinite(artifact[key]).all():
                raise ValueError(f"{name} {key} must be finite")
    for key in ("sample_id", "month", "target"):
        if not np.array_equal(realmlp_npz[key], table_npz[key]):
            raise ValueError(f"RealMLP and Table {key} arrays must align exactly")
    start, end = expected_months
    if set(np.unique(realmlp_npz["month"])) != set(range(start, end + 1)):
        raise ValueError(f"artifact months do not cover registered range {start}-{end}")
    if np.unique(realmlp_npz["sample_id"]).size != realmlp_npz["sample_id"].size:
        raise ValueError("sample_id must be unique")
    return (
        {key: np.asarray(realmlp_npz[key]) for key in required},
        {key: np.asarray(table_npz[key]) for key in required},
    )


def _method_mode(methods: list[str]) -> str:
    counts = Counter(methods)
    return max(METHODS, key=lambda method: (counts[method], -METHODS.index(method)))


def calibrate_clean_baseline(
    realmlp_root: str | Path,
    table_root: str | Path,
    output_root: str | Path,
    *,
    experiment_id: str = "clean-baseline-v2",
) -> dict[str, Any]:
    realmlp_root = Path(realmlp_root)
    table_root = Path(table_root)
    output_root = Path(output_root) / experiment_id
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    methods: list[str] = []
    weights: list[float] = []

    for outer_name, split in NESTED_SPLITS.items():
        realmlp_inner, table_inner = _load_aligned(
            realmlp_root / outer_name / "inner_predictions.npz",
            table_root / outer_name / "inner_predictions.npz",
            expected_months=split.inner_tune.as_tuple(),
        )
        realmlp_outer, table_outer = _load_aligned(
            realmlp_root / outer_name / "predictions.npz",
            table_root / outer_name / "predictions.npz",
            expected_months=split.outer_valid.as_tuple(),
        )
        calibrator = EnsembleCalibrator(METHODS).fit(
            realmlp_inner["pred"], table_inner["pred"], realmlp_inner["target"],
            weight_grid=WEIGHT_GRID,
        )
        assert calibrator.result is not None
        final = calibrator.transform(realmlp_outer["pred"], table_outer["pred"])
        baseline_score = cosine_uncentered(realmlp_outer["pred"], realmlp_outer["target"])
        table_score = cosine_uncentered(table_outer["pred"], table_outer["target"])
        final_score = cosine_uncentered(final, realmlp_outer["target"])
        row = {
            "outer": outer_name,
            "method": calibrator.result.method,
            "table_weight": calibrator.result.weight,
            "inner_score": calibrator.result.score,
            "scale_realmlp": calibrator.result.scale_a,
            "scale_table": calibrator.result.scale_b,
            "realmlp_score": baseline_score,
            "table_score": table_score,
            "final_score": final_score,
            "delta_vs_realmlp": final_score - baseline_score,
            "outer_prediction_corr": float(np.corrcoef(realmlp_outer["pred"], table_outer["pred"])[0, 1]),
            "rows": int(final.size),
            "input_hashes": {
                "realmlp_inner": array_hash(realmlp_inner["pred"]),
                "table_inner": array_hash(table_inner["pred"]),
                "realmlp_outer": array_hash(realmlp_outer["pred"]),
                "table_outer": array_hash(table_outer["pred"]),
            },
        }
        rows.append(row)
        methods.append(calibrator.result.method)
        weights.append(calibrator.result.weight)
        fold_output = output_root / outer_name
        save_predictions(
            fold_output / "predictions.npz", sample_id=realmlp_outer["sample_id"],
            month=realmlp_outer["month"], target=realmlp_outer["target"], pred=final,
            split=np.full(final.size, outer_name),
        )
        (fold_output / "calibration.json").write_text(
            json.dumps(asdict(calibrator.result), indent=2), encoding="utf-8"
        )
    production = {
        "method": _method_mode(methods),
        "table_weight": float(np.median(weights)),
        "scale_source": "canonical_rolling_oof_months_51_70",
        "scale_status": "pending_canonical_oof",
    }
    config_payload = {
        "methods": METHODS,
        "weight_grid": [float(WEIGHT_GRID[0]), float(WEIGHT_GRID[-1]), 0.01],
        "components": ["c2-realmlp-epochs30", "clean-table-v2"],
        "production": production,
    }
    config_hash = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    deltas = [row["delta_vs_realmlp"] for row in rows]
    for row in rows:
        outer_name = row["outer"]
        split = NESTED_SPLITS[outer_name]
        fold_output = output_root / outer_name
        ExperimentManifest(
            experiment_id=f"{experiment_id}-{outer_name.lower()}", status="nested-complete",
            config_hash=config_hash,
            train_months=split.refit_train.as_tuple(), valid_months=split.outer_valid.as_tuple(),
            scores={"cosine_uncentered": row["final_score"]}, diagnostics=row,
        ).write(fold_output)
    gate = {
        "applies_to": "fold_specific_nested_calibrations",
        "all_outers_non_degrading": all(delta >= 0.0 for delta in deltas),
        "mean_delta": float(np.mean(deltas)),
        "required_mean_delta": 0.0005,
    }
    gate["passed"] = bool(gate["all_outers_non_degrading"] and gate["mean_delta"] >= gate["required_mean_delta"])
    if not gate["passed"]:
        raise ValueError(f"Clean Baseline v2 nested gate failed: {gate}")
    report = {
        "experiment_id": experiment_id,
        "status": "calibration-selected-scales-pending",
        "components": {"realmlp": "c2-realmlp-epochs30", "table": "clean-table-v2"},
        "rows": rows,
        "mean_final_score": float(np.mean([row["final_score"] for row in rows])),
        "mean_realmlp_score": float(np.mean([row["realmlp_score"] for row in rows])),
        "mean_table_score": float(np.mean([row["table_score"] for row in rows])),
        "gate": gate,
        "production": production,
        "note": "Outer folds are correlated temporal stress tests, not independent samples.",
    }
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report["result_hash"] = hashlib.sha256(payload).hexdigest()
    (output_root / "clean_baseline_v2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Clean Baseline v2 calibration selection", "",
        "| Outer | Method | Table weight | RealMLP | Table | Final | Delta | Corr |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['outer']} | {row['method']} | {row['table_weight']:.2f} | "
            f"{row['realmlp_score']:.9f} | {row['table_score']:.9f} | {row['final_score']:.9f} | "
            f"{row['delta_vs_realmlp']:+.9f} | {row['outer_prediction_corr']:.4f} |"
        )
    lines += [
        "", f"Mean final score: `{report['mean_final_score']:.9f}`.",
        f"Nested mean delta vs RealMLP: `{gate['mean_delta']:+.9f}` (gate passed).", "",
        f"Production rule: `{production['method']}`, Table weight `{production['table_weight']:.2f}`; "
        "component scales will be fitted on canonical rolling OOF m51-70.", "", report["note"],
    ]
    (output_root / "clean_baseline_v2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
