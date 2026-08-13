"""C4 nested calibration and Clean Baseline v2 production freezing."""

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
from .metrics import cosine_uncentered, rms_scale
from .splits import NESTED_SPLITS


METHODS = ("raw", "std", "rms")
WEIGHT_GRID = np.arange(0.0, 1.0001, 0.01)
PRODUCTION_METHOD = "rms"
PRODUCTION_TABLE_WEIGHT = 0.37
REALMLP_CONFIG_HASH = "de2e26a288361c46016a4462c4eee2201db9d05937f120200297f4a980a79da1"
TABLE_CONFIG_HASH = "02c092c7e8c5012bf49d249f331577e7434c6969a343311ad2be8cb71fc982a9"
CANONICAL_SOURCE_CONTRACTS = {
    "realmlp_m51_60": {
        "experiment_id": "c2-realmlp-epochs30-t3", "config_hash": REALMLP_CONFIG_HASH,
        "train_months": (0, 50), "valid_months": (51, 60),
    },
    "table_m51_60": {
        "experiment_id": "clean-table-v2-t3", "config_hash": TABLE_CONFIG_HASH,
        "train_months": (0, 50), "valid_months": (51, 60),
    },
    "realmlp_m61_70": {
        "experiment_id": "c4-scale-realmlp-r61_70", "config_hash": REALMLP_CONFIG_HASH,
        "train_months": (0, 60), "valid_months": (61, 70),
    },
    "table_m61_70": {
        "experiment_id": "c4-scale-table-r61_70", "config_hash": TABLE_CONFIG_HASH,
        "train_months": (0, 60), "valid_months": (61, 70),
    },
}


def apply_production_rule(
    realmlp_prediction: object,
    table_prediction: object,
    *,
    scale_realmlp: float,
    scale_table: float,
    table_weight: float = 0.37,
) -> np.ndarray:
    """Apply the frozen Clean Baseline v2 directional prediction schema."""

    realmlp = np.asarray(realmlp_prediction, dtype=np.float64).reshape(-1)
    table = np.asarray(table_prediction, dtype=np.float64).reshape(-1)
    if realmlp.shape != table.shape:
        raise ValueError("RealMLP and Table production predictions must have the same shape")
    if not np.isfinite(realmlp).all() or not np.isfinite(table).all():
        raise ValueError("production component predictions must be finite")
    if not 0.0 <= table_weight <= 1.0:
        raise ValueError("Table production weight must be in [0, 1]")
    if not np.isfinite(scale_realmlp) or scale_realmlp <= 0.0:
        raise ValueError("RealMLP production scale must be positive and finite")
    if not np.isfinite(scale_table) or scale_table <= 0.0:
        raise ValueError("Table production scale must be positive and finite")
    return (
        (1.0 - table_weight) * realmlp / scale_realmlp
        + table_weight * table / scale_table
    )


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


def _load_manifested_predictions(
    directory: str | Path,
    contract_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    directory = Path(directory)
    contract = CANONICAL_SOURCE_CONTRACTS[contract_name]
    manifest_path = directory / "manifest.json"
    prediction_path = directory / "predictions.npz"
    if not manifest_path.exists() or not prediction_path.exists():
        raise FileNotFoundError(f"{contract_name}: manifest.json and predictions.npz are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("experiment_id", "config_hash"):
        if manifest.get(key) != contract[key]:
            raise ValueError(
                f"{contract_name} {key} does not match frozen contract: "
                f"{manifest.get(key)!r} != {contract[key]!r}"
            )
    if manifest.get("status") != "complete":
        raise ValueError(f"{contract_name} manifest status must be complete")
    for key in ("train_months", "valid_months"):
        if tuple(manifest.get(key) or ()) != tuple(contract[key]):
            raise ValueError(f"{contract_name} {key} does not match frozen contract")
    with np.load(prediction_path) as source:
        required = {"sample_id", "month", "target", "pred"}
        if not required.issubset(source.files):
            raise ValueError(f"{contract_name} predictions are missing required arrays")
        artifact = {key: np.asarray(source[key]) for key in required}
    expected_months = tuple(contract["valid_months"])
    start, end = expected_months
    if set(np.unique(artifact["month"])) != set(range(start, end + 1)):
        raise ValueError(f"{contract_name} predictions do not cover {start}-{end}")
    if np.unique(artifact["sample_id"]).size != artifact["sample_id"].size:
        raise ValueError(f"{contract_name} sample_id must be unique")
    if not np.isfinite(artifact["target"]).all() or not np.isfinite(artifact["pred"]).all():
        raise ValueError(f"{contract_name} target and predictions must be finite")
    diagnostics = manifest.get("diagnostics") or {}
    manifest_rows = diagnostics.get("n_outer_valid")
    if manifest_rows is None:
        manifest_rows = (diagnostics.get("prediction") or {}).get("count")
    if manifest_rows is None or int(manifest_rows) != artifact["pred"].size:
        raise ValueError(f"{contract_name} manifest row count does not match predictions")
    if not manifest.get("data_fingerprints"):
        raise ValueError(f"{contract_name} manifest must contain data_fingerprints")
    return artifact, manifest


def _assert_component_alignment(
    realmlp: dict[str, np.ndarray], table: dict[str, np.ndarray], block_name: str
) -> None:
    for key in ("sample_id", "month", "target"):
        if not np.array_equal(realmlp[key], table[key]):
            raise ValueError(f"{block_name}: RealMLP and Table {key} arrays must align exactly")


def _gate(deltas: list[float], applies_to: str) -> dict[str, Any]:
    result = {
        "applies_to": applies_to,
        "all_outers_non_degrading": all(delta >= 0.0 for delta in deltas),
        "mean_delta": float(np.mean(deltas)),
        "required_mean_delta": 0.0005,
    }
    result["passed"] = bool(
        result["all_outers_non_degrading"]
        and result["mean_delta"] >= result["required_mean_delta"]
    )
    return result


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
    folds: list[dict[str, Any]] = []
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
        fold_adaptive = calibrator.transform(realmlp_outer["pred"], table_outer["pred"])
        folds.append({
            "outer": outer_name, "split": split, "calibration": calibrator.result,
            "realmlp_inner": realmlp_inner, "table_inner": table_inner,
            "realmlp_outer": realmlp_outer, "table_outer": table_outer,
            "fold_adaptive": fold_adaptive,
        })
        methods.append(calibrator.result.method)
        weights.append(calibrator.result.weight)
    production = {
        "method": PRODUCTION_METHOD,
        "table_weight": PRODUCTION_TABLE_WEIGHT,
        "selected_method_mode": _method_mode(methods),
        "selected_weight_median": float(np.median(weights)),
        "scale_source": "canonical_rolling_oof_months_51_70",
        "scale_status": "pending_canonical_oof",
    }
    rows: list[dict[str, Any]] = []
    for fold in folds:
        outer_name = fold["outer"]
        realmlp_inner = fold["realmlp_inner"]
        table_inner = fold["table_inner"]
        realmlp_outer = fold["realmlp_outer"]
        table_outer = fold["table_outer"]
        fold_adaptive = fold["fold_adaptive"]
        scale_realmlp = rms_scale(realmlp_inner["pred"])
        scale_table = rms_scale(table_inner["pred"])
        production_stress = apply_production_rule(
            realmlp_outer["pred"], table_outer["pred"],
            scale_realmlp=scale_realmlp, scale_table=scale_table,
            table_weight=PRODUCTION_TABLE_WEIGHT,
        )
        baseline_score = cosine_uncentered(realmlp_outer["pred"], realmlp_outer["target"])
        table_score = cosine_uncentered(table_outer["pred"], table_outer["target"])
        fold_adaptive_score = cosine_uncentered(fold_adaptive, realmlp_outer["target"])
        stress_score = cosine_uncentered(production_stress, realmlp_outer["target"])
        calibration = fold["calibration"]
        row = {
            "outer": outer_name,
            "method": calibration.method,
            "table_weight": calibration.weight,
            "inner_score": calibration.score,
            "scale_realmlp": calibration.scale_a,
            "scale_table": calibration.scale_b,
            "realmlp_score": baseline_score,
            "table_score": table_score,
            "final_score": fold_adaptive_score,
            "delta_vs_realmlp": fold_adaptive_score - baseline_score,
            "production_rule_stress_score": stress_score,
            "production_rule_stress_delta_vs_realmlp": stress_score - baseline_score,
            "production_rule_stress_scale_realmlp": scale_realmlp,
            "production_rule_stress_scale_table": scale_table,
            "outer_prediction_corr": float(np.corrcoef(realmlp_outer["pred"], table_outer["pred"])[0, 1]),
            "rows": int(fold_adaptive.size),
            "input_hashes": {
                "realmlp_inner": array_hash(realmlp_inner["pred"]),
                "table_inner": array_hash(table_inner["pred"]),
                "realmlp_outer": array_hash(realmlp_outer["pred"]),
                "table_outer": array_hash(table_outer["pred"]),
            },
        }
        rows.append(row)
        fold_output = output_root / outer_name
        save_predictions(
            fold_output / "fold_adaptive_predictions.npz",
            sample_id=realmlp_outer["sample_id"], month=realmlp_outer["month"],
            target=realmlp_outer["target"], pred=fold_adaptive,
            split=np.full(fold_adaptive.size, f"{outer_name}:fold_adaptive"),
        )
        save_predictions(
            fold_output / "production_rule_stress_predictions.npz",
            sample_id=realmlp_outer["sample_id"], month=realmlp_outer["month"],
            target=realmlp_outer["target"], pred=production_stress,
            split=np.full(production_stress.size, f"{outer_name}:production_rule_stress"),
        )
        np.savez_compressed(
            fold_output / "components.npz",
            sample_id=realmlp_outer["sample_id"], month=realmlp_outer["month"],
            target=realmlp_outer["target"], realmlp_pred=realmlp_outer["pred"],
            table_pred=table_outer["pred"],
        )
        (fold_output / "calibration.json").write_text(
            json.dumps(asdict(calibration), indent=2), encoding="utf-8"
        )
    config_payload = {
        "methods": METHODS,
        "weight_grid": [float(WEIGHT_GRID[0]), float(WEIGHT_GRID[-1]), 0.01],
        "components": ["c2-realmlp-epochs30", "clean-table-v2"],
        "production": production,
    }
    config_hash = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    strict_gate = _gate(
        [row["delta_vs_realmlp"] for row in rows],
        "fold_specific_nested_calibrations",
    )
    stress_gate = _gate(
        [row["production_rule_stress_delta_vs_realmlp"] for row in rows],
        "correlated_post_selection_production_rule_stress",
    )
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
    if not strict_gate["passed"]:
        raise ValueError(f"Clean Baseline v2 nested gate failed: {strict_gate}")
    if not stress_gate["passed"]:
        raise ValueError(f"Clean Baseline v2 production stress gate failed: {stress_gate}")
    report = {
        "experiment_id": experiment_id,
        "status": "calibration-selected-scales-pending",
        "components": {"realmlp": "c2-realmlp-epochs30", "table": "clean-table-v2"},
        "rows": rows,
        "mean_final_score": float(np.mean([row["final_score"] for row in rows])),
        "mean_realmlp_score": float(np.mean([row["realmlp_score"] for row in rows])),
        "mean_table_score": float(np.mean([row["table_score"] for row in rows])),
        "gate": strict_gate,
        "strict_nested_gate": strict_gate,
        "production_rule_stress_gate": stress_gate,
        "production": production,
        "note": "Outer folds are correlated temporal stress tests, not independent samples.",
    }
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report["result_hash"] = hashlib.sha256(payload).hexdigest()
    (output_root / "clean_baseline_v2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Clean Baseline v2 calibration selection", "",
        "| Outer | Method | Table weight | RealMLP | Table | Fold adaptive | Delta | Fixed-rule stress | Stress delta | Corr |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['outer']} | {row['method']} | {row['table_weight']:.2f} | "
            f"{row['realmlp_score']:.9f} | {row['table_score']:.9f} | {row['final_score']:.9f} | "
            f"{row['delta_vs_realmlp']:+.9f} | {row['production_rule_stress_score']:.9f} | "
            f"{row['production_rule_stress_delta_vs_realmlp']:+.9f} | {row['outer_prediction_corr']:.4f} |"
        )
    lines += [
        "", f"Mean final score: `{report['mean_final_score']:.9f}`.",
        f"Nested mean delta vs RealMLP: `{strict_gate['mean_delta']:+.9f}` (gate passed).",
        f"Production-rule stress mean delta: `{stress_gate['mean_delta']:+.9f}` (gate passed).", "",
        f"Production rule: `{production['method']}`, Table weight `{production['table_weight']:.2f}`; "
        "component scales will be fitted on canonical rolling OOF m51-70.", "",
        "The production-rule stress gate is correlated and post-selection; it is not an unbiased outer score.",
        "", report["note"],
    ]
    (output_root / "clean_baseline_v2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def freeze_production_scales(
    realmlp_m51_dir: str | Path,
    table_m51_dir: str | Path,
    realmlp_m61_dir: str | Path,
    table_m61_dir: str | Path,
    calibration_root: str | Path,
    *,
    experiment_id: str = "clean-baseline-v2",
    table_weight: float = PRODUCTION_TABLE_WEIGHT,
) -> dict[str, Any]:
    """Fit the fixed production RMS scales on canonical rolling OOF m51-70."""

    calibration_root = Path(calibration_root)
    calibration_report_path = calibration_root / "clean_baseline_v2_report.json"
    if not calibration_report_path.exists():
        raise FileNotFoundError("calibration root must contain clean_baseline_v2_report.json")
    calibration_report = json.loads(calibration_report_path.read_text(encoding="utf-8"))
    if not calibration_report.get("strict_nested_gate", {}).get("passed"):
        raise ValueError("strict nested calibration gate must pass before freezing")
    if not calibration_report.get("production_rule_stress_gate", {}).get("passed"):
        raise ValueError("production rule stress gate must pass before freezing")

    blocks = []
    source_manifests: dict[str, dict[str, Any]] = {}
    for block_name, realmlp_dir, table_dir, realmlp_contract, table_contract in (
        ("m51_60", realmlp_m51_dir, table_m51_dir, "realmlp_m51_60", "table_m51_60"),
        ("m61_70", realmlp_m61_dir, table_m61_dir, "realmlp_m61_70", "table_m61_70"),
    ):
        realmlp, realmlp_manifest = _load_manifested_predictions(realmlp_dir, realmlp_contract)
        table, table_manifest = _load_manifested_predictions(table_dir, table_contract)
        _assert_component_alignment(realmlp, table, block_name)
        blocks.append((block_name, realmlp, table))
        source_manifests[realmlp_contract] = realmlp_manifest
        source_manifests[table_contract] = table_manifest

    if source_manifests["realmlp_m51_60"]["data_fingerprints"] != source_manifests["realmlp_m61_70"]["data_fingerprints"]:
        raise ValueError("RealMLP canonical blocks must use identical data fingerprints")
    if source_manifests["table_m51_60"]["data_fingerprints"] != source_manifests["table_m61_70"]["data_fingerprints"]:
        raise ValueError("Table canonical blocks must use identical data fingerprints")

    sample_id = np.concatenate([block[1]["sample_id"] for block in blocks])
    month = np.concatenate([block[1]["month"] for block in blocks])
    target = np.concatenate([block[1]["target"] for block in blocks])
    realmlp_pred = np.concatenate([block[1]["pred"] for block in blocks]).astype(np.float64)
    table_pred = np.concatenate([block[2]["pred"] for block in blocks]).astype(np.float64)
    if np.unique(sample_id).size != sample_id.size:
        raise ValueError("canonical rolling OOF sample_id must be unique across m51-70")
    if set(np.unique(month)) != set(range(51, 71)):
        raise ValueError("canonical rolling OOF must cover every month from 51 through 70")
    scale_realmlp = float(np.sqrt(np.mean(np.square(realmlp_pred))))
    scale_table = float(np.sqrt(np.mean(np.square(table_pred))))
    if not np.isfinite(scale_realmlp) or scale_realmlp <= 0.0:
        raise ValueError("RealMLP canonical OOF RMS scale must be positive and finite")
    if not np.isfinite(scale_table) or scale_table <= 0.0:
        raise ValueError("Table canonical OOF RMS scale must be positive and finite")
    prediction = apply_production_rule(
        realmlp_pred,
        table_pred,
        scale_realmlp=scale_realmlp,
        scale_table=scale_table,
        table_weight=table_weight,
    )
    score = cosine_uncentered(prediction, target)
    output = calibration_root / "production"
    save_predictions(
        output / "canonical_scale_predictions.npz",
        sample_id=sample_id,
        month=month,
        target=target,
        pred=prediction,
        split=np.full(prediction.size, "canonical_scale_oof_m51_70"),
    )
    report = {
        "experiment_id": experiment_id,
        "status": "frozen",
        "method": "rms",
        "table_weight": float(table_weight),
        "scale_realmlp": scale_realmlp,
        "scale_table": scale_table,
        "scale_source": "canonical_rolling_oof_months_51_70",
        "canonical_oof_score": float(score),
        "canonical_oof_rows": int(prediction.size),
        "canonical_oof_months": [51, 70],
        "component_scores": {
            "realmlp": cosine_uncentered(realmlp_pred, target),
            "table": cosine_uncentered(table_pred, target),
        },
        "source_hashes": {
            "realmlp_m51_60": array_hash(blocks[0][1]["pred"]),
            "table_m51_60": array_hash(blocks[0][2]["pred"]),
            "realmlp_m61_70": array_hash(blocks[1][1]["pred"]),
            "table_m61_70": array_hash(blocks[1][2]["pred"]),
        },
        "source_manifests": {
            name: {
                key: manifest.get(key)
                for key in (
                    "experiment_id", "git_sha", "config_hash", "feature_hash",
                    "train_months", "valid_months", "data_fingerprints",
                )
            }
            for name, manifest in source_manifests.items()
        },
    }
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report["result_hash"] = hashlib.sha256(payload).hexdigest()
    (output / "production_scales.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    ExperimentManifest(
        experiment_id=f"{experiment_id}-production-scales",
        status="frozen",
        config_hash=report["result_hash"],
        train_months=(0, 60),
        valid_months=(51, 70),
        scores={"cosine_uncentered": float(score)},
        diagnostics=report,
    ).write(output)
    (output / "report.md").write_text(
        "\n".join(
            [
                "# Clean Baseline v2 production scales",
                "",
                f"- method / Table weight: `rms` / `{table_weight:.2f}`",
                f"- RealMLP RMS scale: `{scale_realmlp:.12g}`",
                f"- Table RMS scale: `{scale_table:.12g}`",
                f"- canonical rolling OOF m51-70 score: `{score:.9f}`",
                f"- rows: `{prediction.size}`",
                "",
                "The scales use only canonical rolling OOF predictions and never outer/test distributions.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    production_replay: list[dict[str, Any]] = []
    for outer_name, split in NESTED_SPLITS.items():
        fold_output = calibration_root / outer_name
        component_path = fold_output / "components.npz"
        if not component_path.exists():
            raise FileNotFoundError(f"{outer_name}: calibration components.npz is required")
        with np.load(component_path) as components:
            required = {"sample_id", "month", "target", "realmlp_pred", "table_pred"}
            if not required.issubset(components.files):
                raise ValueError(f"{outer_name}: calibration components are incomplete")
            values = {key: np.asarray(components[key]) for key in required}
        if set(np.unique(values["month"])) != set(range(split.outer_valid.start, split.outer_valid.end + 1)):
            raise ValueError(f"{outer_name}: component months do not match registered outer")
        default_prediction = apply_production_rule(
            values["realmlp_pred"], values["table_pred"],
            scale_realmlp=scale_realmlp, scale_table=scale_table,
            table_weight=table_weight,
        )
        replay_score = cosine_uncentered(default_prediction, values["target"])
        replay = {
            "outer": outer_name,
            "score": replay_score,
            "realmlp_score": cosine_uncentered(values["realmlp_pred"], values["target"]),
            "delta_vs_realmlp": replay_score - cosine_uncentered(values["realmlp_pred"], values["target"]),
            "role": "production_schema_replay_not_a_selection_gate",
        }
        production_replay.append(replay)
        save_predictions(
            fold_output / "predictions.npz", sample_id=values["sample_id"],
            month=values["month"], target=values["target"], pred=default_prediction,
            split=np.full(default_prediction.size, f"{outer_name}:production_default"),
        )
        manifest_path = fold_output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "frozen"
        manifest.setdefault("scores", {})["production_schema_replay_cosine"] = replay_score
        manifest.setdefault("diagnostics", {})["production_schema_replay"] = replay
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    report["production_schema_replay"] = production_replay
    calibration_report.update(
        {
            "status": "frozen",
            "production": {
                "method": PRODUCTION_METHOD,
                "table_weight": float(table_weight),
                "scale_realmlp": scale_realmlp,
                "scale_table": scale_table,
                "scale_source": "canonical_rolling_oof_months_51_70",
                "canonical_result_hash": report["result_hash"],
            },
            "canonical_scale_report": report,
            "production_schema_replay": production_replay,
        }
    )
    calibration_report_path.write_text(
        json.dumps(calibration_report, indent=2), encoding="utf-8"
    )
    return report
