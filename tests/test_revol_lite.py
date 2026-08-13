from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from mscapital.cli import build_parser
from mscapital.features.revol_lite import (
    context_feature_names,
    revol_lite_feature_names,
)
from mscapital.stability import _bootstrap_delta


def _write_feather(path, columns):
    feather.write_feather(pa.table(columns), path)


def test_revol_lite_schema_is_fixed() -> None:
    names = revol_lite_feature_names()
    assert len(names) == 37
    assert len(set(names)) == 37
    assert len(context_feature_names()) == 11
    assert all(name in names for name in context_feature_names())


def test_revol_lite_builder_aligns_rows_and_marks_empty_streams(tmp_path) -> None:
    from mscapital.features.revol_lite import build_revol_lite_file
    from mscapital.models.revol_lite import load_revol_lite_frame

    labels = tmp_path / "label.feather"
    market = tmp_path / "market.feather"
    order = tmp_path / "order.feather"
    trade = tmp_path / "transaction.feather"
    output = tmp_path / "features.parquet"
    _write_feather(labels, {
        "sample_id": np.array([20, 10], dtype=np.int32),
        "month": np.array([22, 21], dtype=np.int16),
        "target": np.array([0.2, -0.1], dtype=np.float32),
    })
    # Rows are oldest-to-newest within each sample (seconds decrease).
    _write_feather(market, {
        "sample_id": np.array([10, 10, 20, 20], dtype=np.int32),
        "seconds_before_predict": np.array([4.0, 0.5, 4.0, 0.5], dtype=np.float32),
        "bid_price_1": np.array([99.0, 99.5, 100.0, 100.0], dtype=np.float64),
        "ask_price_1": np.array([101.0, 101.5, 102.0, 102.0], dtype=np.float64),
        "bid_volume_1": np.ones(4, dtype=np.float64),
        "ask_volume_1": np.ones(4, dtype=np.float64),
        "bid_price_2": np.array([98.0, 98.5, 99.0, 99.0], dtype=np.float64),
        "ask_price_2": np.array([102.0, 102.5, 103.0, 103.0], dtype=np.float64),
        "bid_volume_2": np.ones(4, dtype=np.float64),
        "ask_volume_2": np.ones(4, dtype=np.float64),
    })
    _write_feather(order, {
        "sample_id": np.array([10, 10], dtype=np.int32),
        "seconds_before_predict": np.array([4.0, 0.5], dtype=np.float32),
        "volume": np.array([2.0, 3.0], dtype=np.float64),
        "side": np.array([0, 1], dtype=np.int8),
        "order_action": np.array([0, 0], dtype=np.int8),
    })
    _write_feather(trade, {
        "sample_id": np.array([10], dtype=np.int32),
        "seconds_before_predict": np.array([0.5], dtype=np.float32),
        "volume": np.array([4.0], dtype=np.float64),
        "side": np.array([0], dtype=np.int8),
    })

    result = build_revol_lite_file(market, order, trade, labels, output)
    assert result["feature_count"] == 37
    frame = load_revol_lite_frame(output)
    assert frame.sample_id.tolist() == [10, 20]
    assert frame.values.dtype == np.float32
    market_missing = frame.values[:, frame.feature_names.index("revol_market_missing")]
    order_missing = frame.values[:, frame.feature_names.index("revol_order_missing")]
    trade_missing = frame.values[:, frame.feature_names.index("revol_trade_missing")]
    assert market_missing.tolist() == [0.0, 0.0]
    assert order_missing.tolist() == [0.0, 1.0]
    assert trade_missing.tolist() == [0.0, 1.0]
    assert np.isfinite(frame.values).all()
    manifest = json.loads((output.parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["diagnostics"]["feature_count"] == 37


def test_stability_bootstrap_is_seed_deterministic() -> None:
    rows = [
        {"candidate_dot": 2.0, "candidate_sq": 4.0, "baseline_dot": 1.0, "baseline_sq": 4.0, "target_sq": 4.0},
        {"candidate_dot": 3.0, "candidate_sq": 9.0, "baseline_dot": 2.0, "baseline_sq": 9.0, "target_sq": 9.0},
    ]
    first = _bootstrap_delta(rows, n_bootstrap=200, seed=2026)
    second = _bootstrap_delta(rows, n_bootstrap=200, seed=2026)
    assert first == second


def test_context_bootstrap_and_beta_are_finite_and_deterministic() -> None:
    from mscapital.models.context_shift import _bootstrap_cosine, _fit_beta

    baseline = np.array([1.0, 2.0, 3.0])
    target = np.array([2.0, 4.0, 5.0])
    assert np.isclose(_fit_beta(baseline, target), 25.0 / 14.0)
    rows = [
        {"dot": 2.0, "pred_sq": 1.0, "target_sq": 4.0},
        {"dot": 3.0, "pred_sq": 4.0, "target_sq": 9.0},
    ]
    first = _bootstrap_cosine(rows, seed=2026, n_bootstrap=200)
    second = _bootstrap_cosine(rows, seed=2026, n_bootstrap=200)
    assert first == second
    assert np.isfinite([first["lower_95"], first["upper_95"], first["mean"]]).all()


def test_new_cli_commands_are_registered() -> None:
    parser = build_parser()
    for command in (
        "build-revol-lite", "run-revol-lite", "summarize-revol-lite",
        "audit-candidate-stability", "diagnose-context-shift",
    ):
        parsed = parser.parse_args([command] + {
            "build-revol-lite": ["--market", "m", "--order", "o", "--transaction", "t", "--labels", "l", "--output", "x"],
            "run-revol-lite": ["--canonical-oof", "c", "--features", "f", "--baseline-root", "b", "--output-root", "o", "--outer", "PSEUDO"],
            "summarize-revol-lite": ["--artifact-root", "a", "--output", "o"],
            "audit-candidate-stability": ["--artifact-root", "a", "--features", "f", "--output-root", "o"],
            "diagnose-context-shift": ["--canonical-oof", "c", "--features", "f", "--output-root", "o"],
        }[command])
        assert parsed.command == command
