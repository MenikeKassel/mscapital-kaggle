from __future__ import annotations

import json

import numpy as np
import pytest

from mscapital.artifacts import ExperimentManifest
from mscapital.ensemble import EnsembleCalibrator, NestedBlendFold, evaluate_nested_blend
from mscapital.features.lob_geometry import lob_geometry_row
from mscapital.features.ofi import quote_ofi, signed_order_flow, signed_trade_flow
from mscapital.metrics import cosine_centered, cosine_uncentered
from mscapital.preprocessing import FoldSafeCategoricalEncoder, FoldSafePreprocessor
from mscapital.residual import OOFBlock, build_canonical_oof, outer_residual
from mscapital.splits import NESTED_SPLITS, ROLLING_WINDOWS


def test_uncentered_metric_is_not_centered_and_handles_zero() -> None:
    pred = np.array([2.0, 3.0, 4.0])
    target = np.array([1.0, 2.0, 3.0])
    assert cosine_uncentered(pred, target) != cosine_centered(pred, target)
    assert cosine_uncentered(np.zeros(3), target) == 0.0
    with pytest.raises(ValueError):
        cosine_uncentered([1.0], [1.0, 2.0])


def test_nested_splits_have_no_inner_or_outer_overlap() -> None:
    for split in NESTED_SPLITS.values():
        split.validate()
        assert split.inner_tune.end < split.refit_train.end + 1
    assert [name for name, _, _ in ROLLING_WINDOWS] == [
        "m21_30", "m31_40", "m41_50", "m51_60", "m61_70"
    ]


def test_fold_safe_preprocessor_does_not_learn_validation_extreme() -> None:
    train = np.array([[1.0, np.nan], [2.0, 10.0], [3.0, 20.0]])
    valid = np.array([[1000.0, np.nan], [2.5, 10000.0]])
    pre = FoldSafePreprocessor(["quant", "continuous"], quantile_columns=["quant"], quantile_bins=4)
    train_out = pre.fit_transform(train)
    state_before = [
        (s.median, s.scale, None if s.edges is None else s.edges.copy())
        for s in pre.state
    ]
    valid_out = pre.transform(valid)
    state_after = [
        (s.median, s.scale, None if s.edges is None else s.edges.copy())
        for s in pre.state
    ]
    assert np.array_equal(train_out[:, 0], np.array([0.0, 2.0, 3.0], dtype=np.float32))
    assert valid_out[0, 0] == 3.0
    assert valid_out[0, 1] < 1000.0
    for before, after in zip(state_before, state_after):
        assert before[0] == after[0]
        assert before[1] == after[1]
        if before[2] is not None:
            assert np.array_equal(before[2], after[2])
        else:
            assert after[2] is None


def test_quantile_missing_gets_dedicated_bin() -> None:
    pre = FoldSafePreprocessor(["x"], quantile_columns=["x"], quantile_bins=3)
    pre.fit(np.array([[1.0], [2.0], [3.0]]))
    assert pre.transform(np.array([[np.nan]]))[0, 0] == 3.0


def test_categorical_vocabulary_is_train_only() -> None:
    encoder = FoldSafeCategoricalEncoder().fit(["bid", "ask", "bid"])
    assert encoder.transform(["ask", "future", None]).tolist() == [1, 2, 2]


def test_ofi_sign_convention_and_trade_flow() -> None:
    out = signed_order_flow([1, 1, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1])
    assert out.tolist() == [1.0, -1.0, -1.0, 1.0]
    assert signed_trade_flow([2, 2], [0, 1]).tolist() == [2.0, -2.0]


def test_quote_ofi_movement_and_queue_change() -> None:
    result = quote_ofi(
        bid_price=[100, 101, 101, 100],
        bid_volume=[10, 20, 25, 30],
        ask_price=[102, 102, 101, 101],
        ask_volume=[10, 12, 15, 14],
    )
    assert result[0] == 0.0
    assert result[1] == 20.0 - 2.0
    assert np.isfinite(result).all()


def test_lob_geometry_excludes_l1_relative_price_and_is_invariant() -> None:
    row = lob_geometry_row(99, 10, 101, 20, 98, 5, 102, 15)
    assert "lob_bid1_rel_mid_spread" not in row
    assert "lob_ask1_rel_mid_spread" not in row
    assert row["lob_shape_asymmetry"] != 0.0


def test_canonical_oof_is_unique_and_outer_beta_uses_only_visible_history() -> None:
    block_a = OOFBlock(
        "m21_30_train20", np.array([1, 2]), np.array([21, 22]),
        np.array([2.0, 4.0]), np.array([1.0, 2.0]), 20
    )
    block_b = OOFBlock(
        "m31_40_train30", np.array([3, 4]), np.array([31, 32]),
        np.array([6.0, 8.0]), np.array([3.0, 4.0]), 30
    )
    canonical = build_canonical_oof([block_a, block_b])
    view = outer_residual(canonical, "PSEUDO")
    assert np.asarray(view["sample_id"]).tolist() == [1, 2, 3, 4]
    assert view["beta"] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        build_canonical_oof([
            block_a,
            OOFBlock(
                "dup_train30", np.array([2]), np.array([31]),
                np.array([4.0]), np.array([2.0]), 30
            ),
        ])


def test_ensemble_calibrator_reuses_inner_scales() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0, 30.0])
    y = a + b
    calibrator = EnsembleCalibrator().fit(a, b, y, weight_grid=np.array([0.0, 0.5, 1.0]))
    assert calibrator.result is not None
    transformed = calibrator.transform(a * 100.0, b * 100.0)
    assert np.isfinite(transformed).all()


def test_nested_blend_calibrates_inner_and_scores_outer() -> None:
    result = evaluate_nested_blend([
        NestedBlendFold(
            "T3",
            np.array([1.0, 2.0]), np.array([2.0, 4.0]), np.array([1.0, 2.0]),
            np.array([1.0, 2.0]), np.array([2.0, 4.0]), np.array([1.0, 2.0]),
        )
    ])
    assert result[0]["fold"] == "T3"
    assert np.isfinite(result[0]["outer_score"])


def test_manifest_is_json_serializable(tmp_path) -> None:
    path = ExperimentManifest("test", scores={"cosine": 0.1}).write(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "test"
