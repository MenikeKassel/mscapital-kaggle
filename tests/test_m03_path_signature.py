from __future__ import annotations

import json

import numpy as np
import pytest

from mscapital.artifacts import feature_hash
from mscapital.cli import build_parser
from mscapital.diagnostics import drift_report
from mscapital.features.path_signature import (
    CHANNEL_NAMES,
    PATH_SIGNATURE_FEATURE_COUNT,
    build_path_signature_arrays,
    build_path_signature_file,
    depth2_path_signature,
    path_signature_feature_names,
    path_signature_features_for_rows,
)
from mscapital.metrics import cosine_uncentered
from mscapital.models.m03 import summarize_m03
from mscapital.splits import NESTED_SPLITS


def _market(seconds: np.ndarray) -> dict[str, np.ndarray]:
    seconds = np.asarray(seconds, dtype=float)
    n = seconds.size
    step = np.arange(n, dtype=float)
    return {
        "seconds_before_predict": seconds,
        "bid_price_1": 99.0 + 0.2 * step,
        "ask_price_1": 101.0 + 0.1 * step,
        "bid_price_2": 98.0 + 0.2 * step,
        "ask_price_2": 102.0 + 0.1 * step,
        "bid_volume_1": 10.0 + step,
        "ask_volume_1": 8.0 + 0.5 * step,
        "bid_volume_2": 7.0 + 0.25 * step,
        "ask_volume_2": 9.0 + 0.75 * step,
    }


def _order(seconds: np.ndarray) -> dict[str, np.ndarray]:
    seconds = np.asarray(seconds, dtype=float)
    return {
        "seconds_before_predict": seconds,
        "volume": np.ones(seconds.size),
        "side": np.zeros(seconds.size, dtype=int),
        "order_action": np.zeros(seconds.size, dtype=int),
    }


def _trade(seconds: np.ndarray) -> dict[str, np.ndarray]:
    seconds = np.asarray(seconds, dtype=float)
    return {
        "seconds_before_predict": seconds,
        "volume": np.arange(1, seconds.size + 1, dtype=float),
        "side": np.arange(seconds.size) % 2,
    }


def test_m03_schema_is_exactly_four_times_seven_plus_twenty_one() -> None:
    names = path_signature_feature_names()
    assert CHANNEL_NAMES == (
        "mid_log_return_bps",
        "relative_spread_bps",
        "l1_imbalance",
        "l2_imbalance",
        "normalized_l1_l2_cont_ofi",
        "normalized_signed_trade_volume",
        "normalized_event_clock",
    )
    assert len(names) == PATH_SIGNATURE_FEATURE_COUNT == 112
    assert len(set(names)) == 112


def test_m03_depth_two_signature_matches_hand_computed_area() -> None:
    path = np.zeros((3, 7), dtype=float)
    path[1, 0] = 1.0
    path[2, 0] = 1.0
    path[2, 1] = 1.0
    signature = depth2_path_signature(path)
    assert signature[:7] == pytest.approx([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Counter-clockwise unit right triangle has signed area +1/2.
    assert signature[7] == pytest.approx(0.5)
    assert np.count_nonzero(signature[8:]) == 0


def test_m03_short_path_is_zero_and_uncovered_prefix_is_not_a_fake_point() -> None:
    short = path_signature_features_for_rows(_market(np.array([0.0])), _order(np.array([])), _trade(np.array([])))
    assert np.array_equal(short, np.zeros(112))

    late = path_signature_features_for_rows(
        _market(np.array([2.0, 1.0, 0.0])), _order(np.array([])), _trade(np.array([]))
    )
    assert np.isfinite(late).all()
    assert np.any(late != 0.0)


def test_m03_excludes_future_and_invalid_quote_or_event_rows() -> None:
    clean_market = _market(np.array([60.0, 30.0, 0.0]))
    clean_trade = _trade(np.array([20.0, 1.0]))
    expected = path_signature_features_for_rows(clean_market, _order(np.array([])), clean_trade)

    future_market = {key: np.append(value, 999.0) for key, value in clean_market.items()}
    future_market["seconds_before_predict"][-1] = -1.0
    future_market["bid_price_1"][-1] = 5000.0
    future_market["ask_price_1"][-1] = 5001.0
    future_market["bid_price_2"][-1] = 4999.0
    future_market["ask_price_2"][-1] = 5002.0
    future_trade = {key: np.append(value, 999.0) for key, value in clean_trade.items()}
    future_trade["seconds_before_predict"][-1] = -0.1
    future_trade["side"][-1] = 0
    assert path_signature_features_for_rows(
        future_market, _order(np.array([-1.0])), future_trade
    ) == pytest.approx(expected)

    invalid_market = {key: np.insert(value, 1, value[0]) for key, value in clean_market.items()}
    invalid_market["seconds_before_predict"][1] = 45.0
    invalid_market["ask_price_1"][1] = invalid_market["bid_price_1"][1]
    invalid_trade = {key: np.insert(value, 1, value[0]) for key, value in clean_trade.items()}
    invalid_trade["seconds_before_predict"][1] = 10.0
    invalid_trade["side"][1] = 9
    assert path_signature_features_for_rows(
        invalid_market, _order(np.array([])), invalid_trade
    ) == pytest.approx(expected)


def test_m03_stream_builder_matches_reference_and_is_target_independent(tmp_path, monkeypatch) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.feather as feather
    import pyarrow.parquet as parquet
    import mscapital.features.path_signature as path_signature_module

    # Force both sample groups to cross Arrow batch boundaries.
    monkeypatch.setattr(path_signature_module, "STREAM_BATCH_SIZE", 2)

    market_rows = []
    order_rows = []
    trade_rows = []
    for sample_id, seconds in ((1, np.array([60.0, 30.0, 0.0])), (2, np.array([10.0, 5.0, 0.0]))):
        market = _market(seconds)
        market_rows.append({"sample_id": np.full(seconds.size, sample_id), **market})
        order = _order(seconds[:2])
        order_rows.append({"sample_id": np.full(2, sample_id), **order})
        trade = _trade(seconds[1:])
        trade_rows.append({"sample_id": np.full(2, sample_id), **trade})

    def combine(rows):
        return {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}

    market = combine(market_rows)
    order = combine(order_rows)
    trade = combine(trade_rows)
    market_path, order_path, trade_path = (
        tmp_path / "market.feather", tmp_path / "order.feather", tmp_path / "transaction.feather"
    )
    feather.write_feather(pa.table(market), market_path)
    feather.write_feather(pa.table(order), order_path)
    feather.write_feather(pa.table(trade), trade_path)
    labels_path = tmp_path / "label.feather"
    feather.write_feather(pa.table({
        "sample_id": np.array([1, 2, 3]),
        "month": np.array([21, 21, 22]),
        "target": np.array([0.1, -0.2, 9.0]),
    }), labels_path)

    ids, names, reference = build_path_signature_arrays(
        market, order, trade, sample_ids=np.array([1, 2, 3])
    )
    output = tmp_path / "features" / "path-signature.parquet"
    result = build_path_signature_file(market_path, order_path, trade_path, labels_path, output)
    streamed = parquet.read_table(output)
    streamed_values = np.column_stack([
        streamed[name].to_numpy(zero_copy_only=False) for name in names
    ])
    assert ids.tolist() == [1, 2, 3]
    assert streamed_values == pytest.approx(reference)
    assert result["feature_hash"] == feature_hash(names)
    assert json.loads((output.parent / "manifest.json").read_text())["status"] == "complete"

    changed_labels = tmp_path / "changed-label.feather"
    feather.write_feather(pa.table({
        "sample_id": np.array([1, 2, 3]), "month": np.array([21, 21, 22]),
        "target": np.array([1000.0, -2000.0, 9999.0]),
    }), changed_labels)
    changed_output = tmp_path / "changed" / "path-signature.parquet"
    build_path_signature_file(market_path, order_path, trade_path, changed_labels, changed_output)
    changed = parquet.read_table(changed_output)
    changed_values = np.column_stack([changed[name].to_numpy(zero_copy_only=False) for name in names])
    assert changed_values == pytest.approx(streamed_values)


def test_m03_cli_registers_build_run_and_summary_contracts() -> None:
    parser = build_parser()
    build = parser.parse_args([
        "build-path-signature", "--market", "market.feather", "--order", "order.feather",
        "--transaction", "transaction.feather", "--labels", "label.feather",
        "--output", "path-signature.parquet",
    ])
    assert build.output.as_posix() == "path-signature.parquet"
    run = parser.parse_args([
        "run-m03", "--canonical-oof", "canonical.npz", "--features", "features.parquet",
        "--baseline-root", "baseline", "--output-root", "output", "--outer", "PSEUDO",
    ])
    assert run.config.as_posix() == "configs/m01-a.json"
    summary = parser.parse_args([
        "summarize-m03", "--artifact-root", "output", "--output", "summary.json",
    ])
    assert summary.output.as_posix() == "summary.json"


def test_m03_summary_replays_schema_scores_drift_and_split_labels(tmp_path) -> None:
    for outer in ("PSEUDO", "H2", "T3", "T4"):
        directory = tmp_path / "m03-path-signature" / outer
        directory.mkdir(parents=True)
        valid = NESTED_SPLITS[outer].outer_valid
        month = np.arange(valid.start, valid.end + 1)
        sample_id = np.arange(month.size) + valid.start * 100
        target = np.linspace(-1.0, 1.0, month.size)
        baseline = target + 0.1
        pred = target + 0.05
        inner_pred = np.linspace(-0.5, 0.5, month.size)
        drift = drift_report(inner_pred, pred)
        diagnostics = {
            "outer": outer,
            "baseline_score": cosine_uncentered(baseline, target),
            "final_score": cosine_uncentered(pred, target),
            "delta_vs_baseline": cosine_uncentered(pred, target) - cosine_uncentered(baseline, target),
            "corr_reference": float(np.corrcoef(pred, baseline)[0, 1]),
            "drift": drift,
            "rows": month.size,
        }
        (directory / "manifest.json").write_text(json.dumps({
            "experiment_id": f"m03-path-signature-{outer.lower()}",
            "status": "complete", "diagnostics": diagnostics,
        }))
        np.savez(
            directory / "predictions.npz", sample_id=sample_id, month=month,
            target=target, baseline_pred=baseline, residual_pred=np.zeros(month.size), pred=pred,
            split=np.full(month.size, f"{outer}:m03-path-signature"),
        )
        np.savez(directory / "inner_predictions.npz", pred=inner_pred)

    result = summarize_m03(tmp_path)
    assert result["method"] == "M03 depth-2 Path Signature"
    assert [row["outer"] for row in result["rows"]] == ["PSEUDO", "H2", "T3", "T4"]
    assert result["gate"]["finite_ok"] is True
