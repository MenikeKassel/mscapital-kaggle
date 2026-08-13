import json
from pathlib import Path

import numpy as np

from mscapital.clean_baseline import (
    apply_production_rule,
    calibrate_clean_baseline,
    freeze_production_scales,
)
from mscapital.splits import NESTED_SPLITS, TRAINING_SPLITS


def _write_artifacts(root: Path, *, table: bool, mutate_outer: bool = False) -> None:
    for index, (outer, split) in enumerate(NESTED_SPLITS.items()):
        directory = root / outer
        directory.mkdir(parents=True)
        inner_month = np.arange(split.inner_tune.start, split.inner_tune.end + 1)
        outer_month = np.arange(split.outer_valid.start, split.outer_valid.end + 1)
        inner_target = np.linspace(-1.0, 1.0, inner_month.size)
        outer_target = np.linspace(-1.0, 1.0, outer_month.size)
        if table:
            inner_pred = inner_target + 0.1 * np.sin(np.arange(inner_month.size) + index)
            outer_pred = outer_target + 0.1 * np.sin(np.arange(outer_month.size) + index)
        else:
            inner_pred = inner_target + 0.4 * np.cos(np.arange(inner_month.size) + index)
            outer_pred = outer_target + 0.4 * np.cos(np.arange(outer_month.size) + index)
        if mutate_outer:
            outer_pred = outer_pred * 100.0
        np.savez(
            directory / "inner_predictions.npz", sample_id=np.arange(inner_month.size), month=inner_month,
            target=inner_target, pred=inner_pred,
        )
        np.savez(
            directory / "predictions.npz", sample_id=np.arange(outer_month.size), month=outer_month,
            target=outer_target, pred=outer_pred,
        )


def test_calibration_selection_does_not_depend_on_outer_predictions(tmp_path: Path):
    realmlp = tmp_path / "realmlp"
    table = tmp_path / "table"
    changed = tmp_path / "changed"
    _write_artifacts(realmlp, table=False)
    _write_artifacts(table, table=True)
    _write_artifacts(changed, table=True, mutate_outer=True)
    first = calibrate_clean_baseline(realmlp, table, tmp_path / "first")
    second = calibrate_clean_baseline(realmlp, changed, tmp_path / "second")
    assert [(row["method"], row["table_weight"]) for row in first["rows"]] == [
        (row["method"], row["table_weight"]) for row in second["rows"]
    ]
    assert first["gate"]["applies_to"] == "fold_specific_nested_calibrations"
    assert first["gate"]["passed"] is True
    assert first["production_rule_stress_gate"]["passed"] is True
    assert first["status"] == "calibration-selected-scales-pending"
    for outer in NESTED_SPLITS:
        fold = tmp_path / "first" / "clean-baseline-v2" / outer
        assert (fold / "fold_adaptive_predictions.npz").exists()
        assert (fold / "production_rule_stress_predictions.npz").exists()
        assert (fold / "components.npz").exists()
        assert not (fold / "predictions.npz").exists()


def test_calibration_rejects_misaligned_ids(tmp_path: Path):
    realmlp = tmp_path / "realmlp"
    table = tmp_path / "table"
    _write_artifacts(realmlp, table=False)
    _write_artifacts(table, table=True)
    path = table / "PSEUDO" / "inner_predictions.npz"
    artifact = dict(np.load(path))
    artifact["sample_id"] = artifact["sample_id"][::-1]
    np.savez(path, **artifact)
    with np.testing.assert_raises_regex(ValueError, "sample_id arrays must align"):
        calibrate_clean_baseline(realmlp, table, tmp_path / "out")


def test_canonical_scale_split_is_strictly_historical():
    split = TRAINING_SPLITS["R61_70"]
    assert split.inner_train.as_tuple() == (0, 50)
    assert split.inner_tune.as_tuple() == (51, 60)
    assert split.refit_train.as_tuple() == (0, 60)
    assert split.outer_valid.as_tuple() == (61, 70)
    assert "R61_70" not in NESTED_SPLITS


def _write_scale_block(
    directory: Path,
    months: range,
    pred: np.ndarray,
    target: np.ndarray,
    *,
    experiment_id: str,
    config_hash: str,
    train_months: tuple[int, int],
    data_fingerprints: dict[str, str],
) -> None:
    directory.mkdir(parents=True)
    month = np.asarray(list(months), dtype=np.int16)
    np.savez(
        directory / "predictions.npz",
        sample_id=np.arange(month.size, dtype=np.int64) + int(month[0]) * 100,
        month=month,
        target=target,
        pred=pred,
    )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "status": "complete",
                "config_hash": config_hash,
                "train_months": list(train_months),
                "valid_months": [int(month[0]), int(month[-1])],
                "data_fingerprints": data_fingerprints,
                "diagnostics": {
                    "n_outer_valid": int(month.size),
                    "prediction": {"count": int(month.size)},
                },
            }
        ),
        encoding="utf-8",
    )


def test_freeze_production_scales_uses_exact_canonical_m51_70(tmp_path: Path):
    y51 = np.linspace(-1.0, 1.0, 10)
    y61 = np.linspace(1.0, -1.0, 10)
    paths = [tmp_path / name for name in ("r51", "t51", "r61", "t61")]
    realmlp_hash = "de2e26a288361c46016a4462c4eee2201db9d05937f120200297f4a980a79da1"
    table_hash = "02c092c7e8c5012bf49d249f331577e7434c6969a343311ad2be8cb71fc982a9"
    _write_scale_block(
        paths[0], range(51, 61), y51 * 2.0, y51,
        experiment_id="c2-realmlp-epochs30-t3", config_hash=realmlp_hash,
        train_months=(0, 50), data_fingerprints={"features": "r-data"},
    )
    _write_scale_block(
        paths[1], range(51, 61), y51 * 0.5, y51,
        experiment_id="clean-table-v2-t3", config_hash=table_hash,
        train_months=(0, 50), data_fingerprints={"features": "t-data"},
    )
    _write_scale_block(
        paths[2], range(61, 71), y61 * 4.0, y61,
        experiment_id="c4-scale-realmlp-r61_70", config_hash=realmlp_hash,
        train_months=(0, 60), data_fingerprints={"features": "r-data"},
    )
    _write_scale_block(
        paths[3], range(61, 71), y61 * 0.25, y61,
        experiment_id="c4-scale-table-r61_70", config_hash=table_hash,
        train_months=(0, 60), data_fingerprints={"features": "t-data"},
    )
    realmlp = tmp_path / "cal-realmlp"
    table = tmp_path / "cal-table"
    _write_artifacts(realmlp, table=False)
    _write_artifacts(table, table=True)
    calibrate_clean_baseline(realmlp, table, tmp_path / "calibration")
    calibration_root = tmp_path / "calibration" / "clean-baseline-v2"
    report = freeze_production_scales(*paths, calibration_root)
    expected_rms_r = np.sqrt(np.mean(np.square(np.r_[y51 * 2.0, y61 * 4.0])))
    expected_rms_t = np.sqrt(np.mean(np.square(np.r_[y51 * 0.5, y61 * 0.25])))
    assert report["status"] == "frozen"
    assert report["canonical_oof_months"] == [51, 70]
    assert report["canonical_oof_rows"] == 20
    assert report["scale_realmlp"] == expected_rms_r
    assert report["scale_table"] == expected_rms_t
    combined_y = np.r_[y51, y61]
    expected_pred = (
        0.63 * np.r_[y51 * 2.0, y61 * 4.0] / expected_rms_r
        + 0.37 * np.r_[y51 * 0.5, y61 * 0.25] / expected_rms_t
    )
    expected_score = float(np.dot(expected_pred, combined_y) / (np.linalg.norm(expected_pred) * np.linalg.norm(combined_y)))
    assert report["canonical_oof_score"] == expected_score
    saved = np.load(calibration_root / "production" / "canonical_scale_predictions.npz")
    assert set(saved["month"]) == set(range(51, 71))
    for outer in NESTED_SPLITS:
        fold = calibration_root / outer
        assert (fold / "predictions.npz").exists()
        assert (fold / "fold_adaptive_predictions.npz").exists()
        production = np.load(fold / "predictions.npz")
        assert np.isfinite(production["pred"]).all()
    final_report = json.loads((calibration_root / "clean_baseline_v2_report.json").read_text())
    assert final_report["status"] == "frozen"


def test_freeze_rejects_wrong_canonical_manifest(tmp_path: Path):
    directory = tmp_path / "bad"
    _write_scale_block(
        directory, range(51, 61), np.ones(10), np.ones(10),
        experiment_id="wrong", config_hash="wrong", train_months=(0, 50),
        data_fingerprints={"features": "wrong"},
    )
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "clean_baseline_v2_report.json").write_text(
        json.dumps(
            {
                "strict_nested_gate": {"passed": True},
                "production_rule_stress_gate": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    with np.testing.assert_raises_regex(ValueError, "experiment_id"):
        freeze_production_scales(directory, directory, directory, directory, calibration)


def test_production_rule_schema_rejects_bad_components():
    expected = 0.63 * np.array([1.0, 2.0]) / 2.0 + 0.37 * np.array([3.0, 4.0]) / 4.0
    actual = apply_production_rule(
        [1.0, 2.0], [3.0, 4.0], scale_realmlp=2.0, scale_table=4.0
    )
    np.testing.assert_allclose(actual, expected)
    with np.testing.assert_raises_regex(ValueError, "same shape"):
        apply_production_rule([1.0], [1.0, 2.0], scale_realmlp=1.0, scale_table=1.0)
    with np.testing.assert_raises_regex(ValueError, "must be finite"):
        apply_production_rule([np.nan], [1.0], scale_realmlp=1.0, scale_table=1.0)
