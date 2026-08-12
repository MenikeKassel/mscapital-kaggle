from pathlib import Path

import numpy as np

from mscapital.clean_baseline import calibrate_clean_baseline
from mscapital.splits import NESTED_SPLITS


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
