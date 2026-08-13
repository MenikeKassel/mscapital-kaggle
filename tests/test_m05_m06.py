from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mscapital.models.m05 import STATE_FEATURE_NAMES, _predict, _transform
from mscapital.models.m06 import audit_m06


def test_m05_knn_never_reads_same_or_future_month() -> None:
    state = (np.zeros(2), np.ones(2), np.zeros(2), np.ones(2))
    centers = np.array([[0.0, 0.0], [1.0, 1.0]])
    residual = np.array([10.0, 20.0])
    source_month = np.array([5, 9], dtype=np.int16)
    x = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    pred = _predict(x, np.array([5, 9, 10]), (centers, residual, source_month), state, 8)
    assert pred[0] == 0.0  # no strictly earlier source
    assert pred[1] == 10.0  # month 5 is the only visible source
    assert pred[2] > 0.0  # both historical prototypes are now visible


def test_m05_transform_clips_and_imputes() -> None:
    state = (np.array([-1.0, -2.0]), np.array([1.0, 2.0]), np.array([0.0, 0.5]), np.ones(2))
    out = _transform(np.array([[np.nan, 9.0], [-9.0, 0.5]]), state)
    np.testing.assert_allclose(out, [[0.0, 1.5], [-1.0, 0.0]])


def test_m06_missing_cross_section_is_frozen(tmp_path) -> None:
    train = tmp_path / "train.parquet"
    test = tmp_path / "test.parquet"
    pq.write_table(pa.table({"sample_id": [1, 2], "month": [0, 1], "target": [0.1, 0.2]}), train)
    pq.write_table(pa.table({"sample_id": [3, 4]}), test)
    report = tmp_path / "m06.json"
    result = audit_m06(train, test, report)
    assert result["status"] == "not_identifiable"
    assert result["prediction_artifact"] is None
    assert json.loads(report.read_text(encoding="utf-8"))["reason"].startswith("no reproducible")


def test_m05_state_feature_schema_is_frozen() -> None:
    assert len(STATE_FEATURE_NAMES) == 16
    assert STATE_FEATURE_NAMES[-1] == "x_trans_order_buy_diff_15"
