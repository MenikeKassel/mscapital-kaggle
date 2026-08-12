from __future__ import annotations

import json

import numpy as np
import pytest

from mscapital.artifacts import ExperimentManifest
from mscapital.cli import main
from mscapital.ensemble import EnsembleCalibrator, NestedBlendFold, evaluate_nested_blend
from mscapital.features.lob_geometry import lob_geometry_row
from mscapital.features.ofi import (
    build_m01_features,
    quote_ofi,
    signed_order_flow,
    signed_trade_flow,
)
from mscapital.features.ofi import select_m01_stage
from mscapital.metrics import cosine_centered, cosine_uncentered
from mscapital.preprocessing import (
    FoldSafeCategoricalEncoder,
    FoldSafePreprocessor,
    prepare_target,
)
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


def test_target_rounding_is_explicit() -> None:
    assert prepare_target([0.123456, -0.123456], 4).tolist() == [0.1235, -0.1235]
    assert prepare_target([0.123456], None).tolist() == [0.123456]


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


def test_m01_sorts_seconds_before_from_oldest_to_newest() -> None:
    order = {
        "sample_id": np.array([1]),
        "seconds_before_predict": np.array([0.0]),
        "volume": np.array([1.0]),
        "side": np.array([0]),
        "order_action": np.array([0]),
    }
    transaction = {
        "sample_id": np.array([1]),
        "seconds_before_predict": np.array([0.0]),
        "volume": np.array([1.0]),
        "side": np.array([0]),
    }
    market = {
        "sample_id": np.array([1, 1]),
        "seconds_before_predict": np.array([0.0, 60.0]),
        "bid_price_1": np.array([101.0, 100.0]),
        "bid_volume_1": np.array([20.0, 10.0]),
        "ask_price_1": np.array([103.0, 102.0]),
        "ask_volume_1": np.array([20.0, 10.0]),
        "bid_price_2": np.array([100.0, 99.0]),
        "bid_volume_2": np.array([10.0, 5.0]),
        "ask_price_2": np.array([104.0, 103.0]),
        "ask_volume_2": np.array([10.0, 5.0]),
    }
    _, names, values = build_m01_features(order, transaction, market)
    value = values[0, names.index("quote_ofi_l1_rate_60")]
    # Old quote [100,102] -> new quote [101,103] contributes positive queue.
    assert value > 0.0


def test_m01_stages_are_cumulative() -> None:
    names = [
        "ofi_event_rate_5",
        "quote_ofi_l1_rate_5",
        "quote_ofi_l2_rate_5",
        "ofi_depth_5",
        "ofi_event_rate_fast_slow_5_15",
        "order_trade_gap_15",
    ]
    values = np.arange(len(names), dtype=np.float32)[None, :]
    a_names, _ = select_m01_stage(names, values, "A")
    c_names, _ = select_m01_stage(names, values, "C")
    f_names, _ = select_m01_stage(names, values, "F")
    assert a_names == ["ofi_event_rate_5"]
    assert c_names == ["ofi_event_rate_5", "quote_ofi_l1_rate_5", "quote_ofi_l2_rate_5"]
    assert f_names == names


def test_lob_geometry_excludes_l1_relative_price_and_is_invariant() -> None:
    row = lob_geometry_row(99, 10, 101, 20, 98, 5, 102, 15)
    assert "lob_bid1_rel_mid_spread" not in row
    assert "lob_ask1_rel_mid_spread" not in row
    assert row["lob_shape_asymmetry"] != 0.0


def test_canonical_oof_is_unique_and_outer_beta_uses_only_visible_history() -> None:
    block_a = OOFBlock(
        "m21_30_train20", np.arange(1, 11), np.arange(21, 31),
        np.arange(2.0, 22.0, 2.0), np.arange(1.0, 11.0), 20
    )
    block_b = OOFBlock(
        "m31_40_train30", np.arange(11, 21), np.arange(31, 41),
        np.arange(22.0, 42.0, 2.0), np.arange(11.0, 21.0), 30
    )
    canonical = build_canonical_oof([block_a, block_b])
    view = outer_residual(canonical, "PSEUDO")
    assert np.asarray(view["sample_id"]).tolist() == list(range(1, 13))
    assert view["beta"] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        build_canonical_oof([
            block_a,
            block_b,
            block_b,
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


def test_cli_run_alpha_aligns_ids_and_writes_diagnostics(tmp_path) -> None:
    base = tmp_path / "base.npz"
    residual = tmp_path / "residual.npz"
    target = tmp_path / "target.npz"
    np.savez(base, sample_id=np.array([1, 2, 3]), pred=np.array([1.0, 2.0, 3.0]))
    np.savez(residual, sample_id=np.array([1, 2, 3]), pred=np.array([0.1, 0.2, 0.3]))
    np.savez(target, target=np.array([1.0, 2.0, 3.0]))
    output = tmp_path / "candidate.npz"
    main([
        "run-alpha", "--baseline", str(base), "--residual", str(residual),
        "--target", str(target), "--alpha", "0.1", "--output", str(output),
    ])
    assert output.exists()
    assert json.loads(output.with_suffix(".json").read_text())["prediction"]["finite"] == 3
