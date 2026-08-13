"""Command line entry points for protocol-v2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .artifacts import ExperimentManifest, save_predictions
from .config import config_hash, load_config
from .features.lob_geometry import build_lob_geometry
from .features.ofi import build_m01_features, select_m01_stage
from .metrics import cosine_uncentered, normalize_prediction
from .diagnostics import prediction_diagnostics, drift_report
from .residual import (
    OOFBlock,
    build_canonical_oof,
    build_clean_baseline_oof_block,
    load_clean_baseline_oof_block,
    outer_residual,
    rolling_window_spec,
)
from .splits import NESTED_SPLITS, OUTER_SPLITS, TRAINING_SPLITS


def _load_json_mapping(path: Path | None) -> dict:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_data_path(value: str | Path | None, data_root: Path, fallback: str) -> Path:
    path = Path(value if value is not None else fallback)
    return path if path.is_absolute() else data_root / path


def _read_arrow(path: str | Path) -> dict[str, np.ndarray]:
    try:
        import pyarrow.feather as feather
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to read Feather event files") from exc
    table = feather.read_table(path, memory_map=True)
    return {name: column.to_numpy(zero_copy_only=False) for name, column in zip(table.column_names, table.columns)}


def _cmd_splits(_args: argparse.Namespace) -> None:
    payload = {
        "outer": {
            key: {"train": value.train.as_tuple(), "valid": value.valid.as_tuple()}
            for key, value in OUTER_SPLITS.items()
        },
        "nested": {
            key: {
                "inner_train": value.inner_train.as_tuple(),
                "inner_tune": value.inner_tune.as_tuple(),
                "refit_train": value.refit_train.as_tuple(),
                "outer_valid": value.outer_valid.as_tuple(),
            }
            for key, value in NESTED_SPLITS.items()
        },
        "rolling_windows": rolling_window_spec(),
    }
    print(json.dumps(payload, indent=2))


def _cmd_score(args: argparse.Namespace) -> None:
    data = np.load(args.predictions)
    pred = data[args.pred_key]
    target = data[args.target_key]
    score = cosine_uncentered(pred, target)
    print(json.dumps({"metric": "cosine_uncentered", "score": score}, indent=2))


def _cmd_clean_baseline(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    experiment_id = args.experiment_id
    directory = cfg.artifact_root / experiment_id
    manifest = ExperimentManifest(
        experiment_id=experiment_id,
        status="scaffold-validated",
        protocol=cfg.protocol,
        config_hash=config_hash(cfg),
        diagnostics={
            "outer_splits": list(OUTER_SPLITS),
            "nested_calibration": True,
            "metric": "cosine_uncentered",
            "submission_enabled": False,
        },
    )
    manifest.write(directory)
    (directory / "report.md").write_text(
        "# Clean Baseline v2\n\nProtocol validated. Training is intentionally explicit and does not submit to Kaggle.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": manifest.status, "manifest": str(directory / "manifest.json")}, indent=2))


def _cmd_clean_realmlp(args: argparse.Namespace) -> None:
    from .models.realmlp import RealMLPConfig, load_frame, run_outer

    mapping = _load_json_mapping(args.config)
    data_root = Path(os.environ.get("MSCAP_DATA_ROOT", mapping.get("data_root", ".")))
    train_path = _resolve_data_path(args.train_path or mapping.get("train_path"), data_root, "processed/f0726_train_f32.parquet")
    labels_path = _resolve_data_path(args.labels_path or mapping.get("labels_path"), data_root, "raw/train/label.feather")
    artifact_root = Path(args.artifact_root or os.environ.get("MSCAP_ARTIFACT_ROOT", mapping.get("artifact_root", "output/experiments")))
    experiment_id = args.experiment_id
    model_config = RealMLPConfig.from_mapping(mapping)
    overrides = {}
    if args.device:
        overrides["device"] = args.device
    if args.max_rows_per_month is not None:
        overrides["max_rows_per_month"] = args.max_rows_per_month
    if overrides:
        model_config = RealMLPConfig(**{**model_config.__dict__, **overrides})
    frame = load_frame(train_path, labels_path, max_rows_per_month=model_config.max_rows_per_month)
    legacy_path = args.legacy_pseudo or mapping.get("legacy_pseudo_path")
    if legacy_path is not None:
        legacy_path = Path(legacy_path)
    selected = tuple(NESTED_SPLITS) if args.outer == "ALL" else (args.outer,)
    results = []
    for outer in selected:
        result = run_outer(
            frame,
            outer,
            model_config,
            artifact_root / experiment_id,
            experiment_id=experiment_id,
            data_paths=(train_path, labels_path),
            legacy_pseudo_path=legacy_path,
        )
        results.append({"outer": outer, "score": result["diagnostics"]["cosine_uncentered"], "artifact": str(artifact_root / experiment_id / outer)})
    print(json.dumps({"status": "complete", "results": results}, indent=2))


def _cmd_summarize_clean_realmlp(args: argparse.Namespace) -> None:
    from .models.realmlp import summarize_outer

    report = summarize_outer(args.artifact_root, args.experiment_id, args.legacy_pseudo)
    report_name = f"{args.experiment_id.replace('-', '_')}_report.json"
    print(json.dumps({"status": "complete", "mean_score": report["mean_score"], "report": str(Path(args.artifact_root) / report_name)}, indent=2))


def _cmd_clean_table(args: argparse.Namespace) -> None:
    from .models.clean_table import CleanTableConfig, load_table_frame, run_outer

    mapping = _load_json_mapping(args.config)
    data_root = Path(os.environ.get("MSCAP_DATA_ROOT", mapping.get("data_root", ".")))
    features_path = _resolve_data_path(args.features_path or mapping.get("features_path"), data_root, "processed/train_features.parquet")
    micro_path = _resolve_data_path(args.micro_path or mapping.get("micro_path"), data_root, "processed/micro_features_train.parquet")
    labels_path = _resolve_data_path(args.labels_path or mapping.get("labels_path"), data_root, "raw/train/label.feather")
    artifact_root = Path(args.artifact_root or os.environ.get("MSCAP_ARTIFACT_ROOT", mapping.get("artifact_root", "output/experiments")))
    config = CleanTableConfig.from_mapping(mapping)
    overrides = {}
    if args.device:
        overrides["device"] = args.device
    if args.max_rows_per_month is not None:
        overrides["max_rows_per_month"] = args.max_rows_per_month
    if overrides:
        config = CleanTableConfig(**{**config.__dict__, **overrides})
    frame = load_table_frame(
        features_path, micro_path, labels_path, max_rows_per_month=config.max_rows_per_month
    )
    selected = tuple(NESTED_SPLITS) if args.outer == "ALL" else (args.outer,)
    results = [
        run_outer(
            frame, outer, config, artifact_root / args.experiment_id,
            experiment_id=args.experiment_id,
            data_paths=(features_path, micro_path, labels_path),
            legacy_pseudo_path=args.legacy_pseudo or mapping.get("legacy_pseudo_path"),
        )
        for outer in selected
    ]
    print(json.dumps({"status": "complete", "results": results}, indent=2))


def _cmd_summarize_clean_table(args: argparse.Namespace) -> None:
    from .models.clean_table import summarize_outer

    report = summarize_outer(args.artifact_root, args.experiment_id, args.legacy_pseudo)
    print(json.dumps({"status": "complete", "mean_score": report["mean_score"]}, indent=2))


def _cmd_verify_legacy_table(args: argparse.Namespace) -> None:
    from .models.clean_table import verify_legacy_anchor

    result = verify_legacy_anchor(args.table_pseudo, args.realmlp_pseudo)
    print(json.dumps(result, indent=2))


def _cmd_calibrate_clean_baseline(args: argparse.Namespace) -> None:
    from .clean_baseline import calibrate_clean_baseline

    result = calibrate_clean_baseline(
        args.realmlp_root, args.table_root, args.output_root, experiment_id=args.experiment_id
    )
    print(json.dumps({"status": result["status"], "gate": result["gate"], "production": result["production"]}, indent=2))


def _cmd_freeze_clean_baseline(args: argparse.Namespace) -> None:
    from .clean_baseline import freeze_production_scales

    result = freeze_production_scales(
        args.realmlp_m51_dir,
        args.table_m51_dir,
        args.realmlp_m61_dir,
        args.table_m61_dir,
        args.calibration_root,
        experiment_id=args.experiment_id,
    )
    print(json.dumps(result, indent=2))


def _cmd_realmlp_inner(args: argparse.Namespace) -> None:
    from .models.realmlp import RealMLPConfig, load_frame, run_inner_diagnostic

    mapping = _load_json_mapping(args.config)
    data_root = Path(os.environ.get("MSCAP_DATA_ROOT", mapping.get("data_root", ".")))
    train_path = _resolve_data_path(args.train_path or mapping.get("train_path"), data_root, "processed/f0726_train_f32.parquet")
    labels_path = _resolve_data_path(args.labels_path or mapping.get("labels_path"), data_root, "raw/train/label.feather")
    artifact_root = Path(args.artifact_root or os.environ.get("MSCAP_ARTIFACT_ROOT", mapping.get("artifact_root", "output/experiments")))
    config = RealMLPConfig.from_mapping(mapping)
    overrides = {}
    if args.device:
        overrides["device"] = args.device
    if args.max_rows_per_month is not None:
        overrides["max_rows_per_month"] = args.max_rows_per_month
    if overrides:
        config = RealMLPConfig(**{**config.__dict__, **overrides})
    frame = load_frame(train_path, labels_path, max_rows_per_month=config.max_rows_per_month)
    selected = ("PSEUDO", "H2", "T3") if args.outer == "ALL" else (args.outer,)
    results = [
        run_inner_diagnostic(
            frame,
            outer,
            config,
            artifact_root / args.experiment_id,
            experiment_id=args.experiment_id,
            data_paths=(train_path, labels_path),
        )
        for outer in selected
    ]
    print(json.dumps({"status": "complete", "results": results}, indent=2))


def _cmd_compare_realmlp(args: argparse.Namespace) -> None:
    from .models.realmlp import compare_outer_experiments

    report = compare_outer_experiments(args.artifact_root, args.baseline_id, args.candidate_id)
    print(json.dumps({"status": "complete", "gate": report["gate"]}, indent=2))


def _cmd_compare_realmlp_inner(args: argparse.Namespace) -> None:
    from .models.realmlp import compare_inner_diagnostics

    report = compare_inner_diagnostics(args.artifact_root, args.baseline_id, args.candidate_id)
    print(json.dumps({"status": "complete", "gate": report["gate"]}, indent=2))


def _parse_block(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("block must be NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def _cmd_build_clean_baseline_oof_block(args: argparse.Namespace) -> None:
    result = build_clean_baseline_oof_block(
        args.realmlp_dir, args.table_dir, args.split, args.output_root,
        allow_smoke_config=args.smoke,
    )
    print(json.dumps(result, indent=2))


def _cmd_build_residual_oof(args: argparse.Namespace) -> None:
    blocks = [
        load_clean_baseline_oof_block(path, split_name)
        for split_name, path in map(_parse_block, args.block)
    ]
    canonical = build_canonical_oof(blocks)
    canonical.save(args.output)
    result = {"output": str(args.output), "rows": int(canonical.sample_id.size)}
    if args.outer:
        view = outer_residual(canonical, args.outer)
        result["outer"] = args.outer
        result["beta"] = float(view["beta"])
        result["rows_visible"] = int(np.asarray(view["sample_id"]).size)
    print(json.dumps(result, indent=2))


def _cmd_build_ofi(args: argparse.Namespace) -> None:
    order, transaction, market = (_read_arrow(path) for path in (args.order, args.transaction, args.market))
    ids, names, values = build_m01_features(order, transaction, market)
    if args.stage:
        names, values = select_m01_stage(names, values, args.stage)
    save_predictions(args.output, sample_id=ids, pred=values)
    metadata = Path(args.output).with_suffix(".json")
    metadata.write_text(json.dumps({"feature_names": names, "n_rows": int(ids.size)}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": int(ids.size), "features": len(names)}, indent=2))


def _cmd_run_alpha(args: argparse.Namespace) -> None:
    baseline = np.load(args.baseline)
    residual = np.load(args.residual)
    base = np.asarray(baseline[args.pred_key], dtype=float).reshape(-1)
    alpha_pred = np.asarray(residual[args.pred_key], dtype=float).reshape(-1)
    if base.shape != alpha_pred.shape:
        raise ValueError("baseline and residual predictions must have equal shapes")
    if not np.isfinite(base).all() or not np.isfinite(alpha_pred).all():
        raise ValueError("baseline and residual predictions must be finite")
    if args.id_key in baseline and args.id_key in residual:
        base_ids = np.asarray(baseline[args.id_key]).reshape(-1)
        residual_ids = np.asarray(residual[args.id_key]).reshape(-1)
        if not np.array_equal(base_ids, residual_ids):
            raise ValueError("baseline and residual sample_id arrays must be identically aligned")
    target = None
    if args.target:
        target = np.asarray(np.load(args.target)[args.target_key], dtype=float).reshape(-1)
        if target.shape != base.shape:
            raise ValueError("target must match prediction shape")
    b, _ = normalize_prediction(base, "rms")
    r, _ = normalize_prediction(alpha_pred, "rms")
    final = b + float(args.alpha) * r
    ids = np.asarray(baseline[args.id_key]) if args.id_key in baseline else np.arange(base.size)
    save_predictions(args.output, sample_id=ids, pred=final, target=target)
    report = prediction_diagnostics(final, target) if target is not None else prediction_diagnostics(final)
    if args.valid:
        valid = np.asarray(np.load(args.valid)[args.pred_key], dtype=float)
        report["drift"] = drift_report(valid, final)
    Path(args.output).with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _cmd_analyze_lb142(args: argparse.Namespace) -> None:
    arrays = []
    labels = []
    for item in args.member:
        label, path = _parse_block(item)
        data = np.load(path)
        arrays.append(np.asarray(data[args.pred_key], dtype=float).reshape(-1))
        labels.append(label)
    if not arrays:
        raise ValueError("at least one LB142 member is required")
    if len({row.size for row in arrays}) != 1:
        raise ValueError("LB142 members must be finite and have equal lengths")
    matrix = np.vstack(arrays)
    if not np.isfinite(matrix).all():
        raise ValueError("LB142 members must be finite and have equal lengths")
    corr = np.corrcoef(matrix)
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    _, singular, _ = np.linalg.svd(centered, full_matrices=False)
    explained = (singular ** 2) / max(float(np.sum(singular ** 2)), 1e-12)
    result = {
        "members": labels,
        "pairwise_pearson": corr.tolist(),
        "pca_explained": explained.tolist(),
        "effective_rank_entropy": float(np.exp(-np.sum(explained * np.log(np.maximum(explained, 1e-12))))),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def _cmd_build_geometry(args: argparse.Namespace) -> None:
    market = _read_arrow(args.market)
    ids, names, values = build_lob_geometry(market)
    save_predictions(args.output, sample_id=ids, pred=values)
    Path(args.output).with_suffix(".json").write_text(json.dumps({"feature_names": names}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": int(ids.size), "features": len(names)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mscapital")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("splits", help="print the registered temporal splits")
    p.set_defaults(func=_cmd_splits)
    p = sub.add_parser("score", help="score a prediction NPZ")
    p.add_argument("predictions", type=Path)
    p.add_argument("--pred-key", default="pred")
    p.add_argument("--target-key", default="target")
    p.set_defaults(func=_cmd_score)
    p = sub.add_parser("clean-baseline", help="validate and register a clean protocol run")
    p.add_argument("--config", type=Path)
    p.add_argument("--experiment-id", default="clean-baseline-v2")
    p.set_defaults(func=_cmd_clean_baseline)
    p = sub.add_parser("clean-realmlp", help="run one or all nested Clean RealMLP-v2a outer folds")
    p.add_argument("--config", type=Path)
    p.add_argument("--outer", choices=tuple(TRAINING_SPLITS) + ("ALL",), required=True)
    p.add_argument("--experiment-id", default="clean-realmlp-v2a")
    p.add_argument("--train-path", type=Path)
    p.add_argument("--labels-path", type=Path)
    p.add_argument("--legacy-pseudo", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--device")
    p.add_argument("--max-rows-per-month", type=int)
    p.set_defaults(func=_cmd_clean_realmlp)
    p = sub.add_parser("summarize-clean-realmlp", help="summarize four completed Clean RealMLP outer folds")
    p.add_argument("--artifact-root", type=Path, default=Path("output/experiments"))
    p.add_argument("--experiment-id", default="clean-realmlp-v2a")
    p.add_argument("--legacy-pseudo", type=Path)
    p.set_defaults(func=_cmd_summarize_clean_realmlp)
    p = sub.add_parser("clean-table", help="run one or all nested Clean Table v2 outer folds")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--outer", choices=tuple(TRAINING_SPLITS) + ("ALL",), required=True)
    p.add_argument("--experiment-id", default="clean-table-v2")
    p.add_argument("--features-path", type=Path)
    p.add_argument("--micro-path", type=Path)
    p.add_argument("--labels-path", type=Path)
    p.add_argument("--legacy-pseudo", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--device")
    p.add_argument("--max-rows-per-month", type=int)
    p.set_defaults(func=_cmd_clean_table)
    p = sub.add_parser("summarize-clean-table", help="summarize four completed Clean Table outer folds")
    p.add_argument("--artifact-root", type=Path, default=Path("output/experiments"))
    p.add_argument("--experiment-id", default="clean-table-v2")
    p.add_argument("--legacy-pseudo", type=Path)
    p.set_defaults(func=_cmd_summarize_clean_table)
    p = sub.add_parser("verify-legacy-table", help="recompute the three frozen v5/v7 PSEUDO anchors")
    p.add_argument("--table-pseudo", type=Path, required=True)
    p.add_argument("--realmlp-pseudo", type=Path, required=True)
    p.set_defaults(func=_cmd_verify_legacy_table)
    p = sub.add_parser("calibrate-clean-baseline", help="run C4 nested RealMLP/Table calibration")
    p.add_argument("--realmlp-root", type=Path, required=True)
    p.add_argument("--table-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--experiment-id", default="clean-baseline-v2")
    p.set_defaults(func=_cmd_calibrate_clean_baseline)
    p = sub.add_parser("freeze-clean-baseline", help="fit production RMS scales from canonical rolling OOF m51-70")
    p.add_argument("--realmlp-m51-dir", type=Path, required=True)
    p.add_argument("--table-m51-dir", type=Path, required=True)
    p.add_argument("--realmlp-m61-dir", type=Path, required=True)
    p.add_argument("--table-m61-dir", type=Path, required=True)
    p.add_argument("--calibration-root", type=Path, required=True)
    p.add_argument("--experiment-id", default="clean-baseline-v2")
    p.set_defaults(func=_cmd_freeze_clean_baseline)
    p = sub.add_parser("realmlp-inner", help="run a C2 inner-only RealMLP diagnostic")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--outer", choices=("PSEUDO", "H2", "T3", "ALL"), required=True)
    p.add_argument("--experiment-id", default="c2-realmlp-ceiling")
    p.add_argument("--train-path", type=Path)
    p.add_argument("--labels-path", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--device")
    p.add_argument("--max-rows-per-month", type=int)
    p.set_defaults(func=_cmd_realmlp_inner)
    p = sub.add_parser("compare-realmlp", help="compare a C2 four-fold candidate with C1")
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--baseline-id", default="clean-realmlp-v2a")
    p.add_argument("--candidate-id", required=True)
    p.set_defaults(func=_cmd_compare_realmlp)
    p = sub.add_parser("compare-realmlp-inner", help="apply the C2 three-inner screening gate")
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--baseline-id", default="clean-realmlp-v2a")
    p.add_argument("--candidate-id", required=True)
    p.set_defaults(func=_cmd_compare_realmlp_inner)
    p = sub.add_parser("build-clean-baseline-oof-block", help="build one fixed-rule rolling OOF block")
    p.add_argument("--realmlp-dir", type=Path, required=True)
    p.add_argument("--table-dir", type=Path, required=True)
    p.add_argument("--split", choices=("R21_30", "R31_40", "R41_50", "R51_60", "R61_70"), required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument(
        "--smoke", action="store_true",
        help="allow noncanonical component configs and mark the output as smoke-only",
    )
    p.set_defaults(func=_cmd_build_clean_baseline_oof_block)
    p = sub.add_parser("build-residual-oof", help="merge unique rolling OOF blocks")
    p.add_argument(
        "--block", action="append", required=True,
        help="SPLIT=PATH to a formal rolling block directory or its predictions.npz",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--outer", choices=tuple(OUTER_SPLITS))
    p.set_defaults(func=_cmd_build_residual_oof)
    p = sub.add_parser("build-ofi", help="build M01 OFI features from Feather files")
    p.add_argument("--order", type=Path, required=True)
    p.add_argument("--transaction", type=Path, required=True)
    p.add_argument("--market", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=("A", "B", "C", "D", "E", "F"))
    p.set_defaults(func=_cmd_build_ofi)
    p = sub.add_parser("build-geometry", help="build market-centered LOB geometry")
    p.add_argument("--market", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=_cmd_build_geometry)
    p = sub.add_parser("run-alpha", help="combine an RMS baseline with a residual prediction")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--residual", type=Path, required=True)
    p.add_argument("--target", type=Path)
    p.add_argument("--valid", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--pred-key", default="pred")
    p.add_argument("--target-key", default="target")
    p.add_argument("--id-key", default="sample_id")
    p.set_defaults(func=_cmd_run_alpha)
    p = sub.add_parser("analyze-lb142", help="PCA/correlation forensic report for prediction members")
    p.add_argument("--member", action="append", required=True, help="NAME=NPZ")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pred-key", default="pred")
    p.set_defaults(func=_cmd_analyze_lb142)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
