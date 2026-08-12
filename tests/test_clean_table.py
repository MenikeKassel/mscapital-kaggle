import json
from pathlib import Path

import numpy as np

from mscapital.models import clean_table
from mscapital.models.clean_table import (
    CleanTableConfig,
    R2_REPLACEMENTS,
    StandardClip,
    TableFrame,
    apply_r2,
    run_outer,
)
from scripts.build_kaggle_c3 import _kernel_code, _package_payload


def test_r2_replaces_exactly_eight_directional_features():
    values = {}
    for output, (numerator, denominator, _offset) in R2_REPLACEMENTS.items():
        values[numerator] = np.array([2.0, 6.0])
        values[denominator] = np.array([1.0, 2.0])
    values["untouched"] = np.array([7.0, 8.0])
    result = apply_r2(values)
    assert len(R2_REPLACEMENTS) == 8
    np.testing.assert_array_equal(result["untouched"], values["untouched"])
    for output, (numerator, denominator, offset) in R2_REPLACEMENTS.items():
        np.testing.assert_allclose(result[output], values[numerator] / (values[denominator] + offset))


def test_standard_clip_state_is_train_only_and_handles_missing():
    train = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    valid = np.array([[1e9, -1e9]], dtype=np.float32)
    state = StandardClip().fit(train)
    before = state.state_hash
    transformed = state.transform(valid)
    assert state.state_hash == before
    assert np.isfinite(transformed).all()
    assert np.max(np.abs(transformed)) <= 10.0


def _small_frame() -> TableFrame:
    months = np.repeat(np.arange(71), 2).astype(np.int16)
    sample_id = np.arange(months.size, dtype=np.int64)
    values = np.column_stack((sample_id / 10.0, np.sin(sample_id))).astype(np.float32)
    target = (0.01 * np.cos(sample_id)).astype(np.float32)
    return TableFrame(sample_id, months, target, values, ("a", "b"))


def test_outer_valid_mutation_does_not_change_inner_or_refit_state(monkeypatch, tmp_path: Path):
    def lgb_inner(x, y, xv, yv, cfg):
        return xv[:, 0], 2, {"valid": [0.1, 0.2]}

    def cat_inner(x, y, xv, yv, cfg):
        return xv[:, 1], 3, {"validation": [0.1, 0.2, 0.3]}

    def lgb_refit(x, y, xv, steps, cfg):
        return xv[:, 0] * steps

    def cat_refit(x, y, xv, steps, cfg):
        return xv[:, 1] * steps

    def mlp(x, y, xv, cfg, seed, *, yv=None, epochs=None):
        step = 4 if epochs is None else epochs
        return xv[:, 0] + seed * 1e-6, step, [{"epoch": 1, "loss": 1.0}]

    monkeypatch.setattr(clean_table, "_fit_lgb_inner", lgb_inner)
    monkeypatch.setattr(clean_table, "_fit_cat_inner", cat_inner)
    monkeypatch.setattr(clean_table, "_fit_lgb_refit", lgb_refit)
    monkeypatch.setattr(clean_table, "_fit_cat_refit", cat_refit)
    monkeypatch.setattr(clean_table, "_train_mlp_seed", mlp)
    cfg = CleanTableConfig(mlp_seeds=(7,), lgb_iterations=2, cat_iterations=3, mlp_epochs=4)
    frame = _small_frame()
    first = run_outer(frame, "T4", cfg, tmp_path / "first")
    changed = TableFrame(
        frame.sample_id, frame.month, np.where(frame.month >= 61, frame.target + 10, frame.target),
        np.where((frame.month >= 61)[:, None], frame.values + 1000, frame.values), frame.feature_names,
    )
    second = run_outer(changed, "T4", cfg, tmp_path / "second")
    assert first["best_steps"] == second["best_steps"]
    m1 = json.loads((tmp_path / "first" / "T4" / "manifest.json").read_text())
    m2 = json.loads((tmp_path / "second" / "T4" / "manifest.json").read_text())
    assert m1["diagnostics"]["mlp_inner_scaler_hash"] == m2["diagnostics"]["mlp_inner_scaler_hash"]
    assert m1["diagnostics"]["mlp_refit_scaler_hash"] == m2["diagnostics"]["mlp_refit_scaler_hash"]
    inner1 = np.load(tmp_path / "first" / "T4" / "inner_predictions.npz")
    inner2 = np.load(tmp_path / "second" / "T4" / "inner_predictions.npz")
    np.testing.assert_array_equal(inner1["pred"], inner2["pred"])


def test_legacy_verifier_rejects_target_misalignment(tmp_path: Path):
    table = tmp_path / "table.npz"
    realmlp = tmp_path / "realmlp.npz"
    np.savez(table, pred=np.array([1.0, 2.0]), y=np.array([1.0, 2.0]))
    np.savez(realmlp, pred=np.array([1.0, 2.0]), y=np.array([2.0, 1.0]))
    with np.testing.assert_raises_regex(ValueError, "targets do not match"):
        clean_table.verify_legacy_anchor(table, realmlp)


def test_kaggle_payload_contains_model_and_mount_preflight():
    repo = Path(__file__).resolve().parents[1]
    payload = _package_payload(repo / "src" / "mscapital")
    code = _kernel_code(payload, "PSEUDO", "abc123", {}, "clean-table-v2")
    assert "models/clean_table.py" not in code  # compressed, not a filesystem import
    assert "train_features.parquet" in code
    assert "micro_features_train.parquet" in code
    assert "label.feather" in code
    assert 'MSCAP_GIT_SHA"] = "abc123"' in code
