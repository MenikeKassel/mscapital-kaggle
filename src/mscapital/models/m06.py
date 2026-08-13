"""Cross-sectional Dynamics schema audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def audit_m06(train_path: str | Path, test_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Prove whether train/test expose a deployable asset/time cross-section."""
    import pyarrow.parquet as pq
    import pyarrow.feather as feather

    def names(path: Path) -> list[str]:
        if path.suffix.lower() == ".parquet":
            return list(pq.read_schema(path).names)
        return list(feather.read_table(path, memory_map=True).schema.names)

    train = Path(train_path); test = Path(test_path)
    train_names = names(train); test_names = names(test)
    candidates = ["asset_id", "stock_id", "instrument_id", "time_id", "timestamp", "date"]
    shared = sorted(set(train_names) & set(test_names) & set(candidates))
    has_asset = any(x in shared for x in ("asset_id", "stock_id", "instrument_id"))
    has_time = any(x in shared for x in ("time_id", "timestamp", "date"))
    if has_asset and has_time:
        status = "identifiable"
        reason = "explicit shared asset/time keys exist; downstream historical-only implementation required"
    else:
        status = "not_identifiable"
        reason = "no reproducible asset/time cross-section in train and test"
    result = {"method": "M06 Cross-sectional Dynamics", "status": status, "reason": reason, "train_columns": train_names, "test_columns": test_names, "shared_candidate_keys": shared, "prediction_artifact": None}
    output = Path(output_path); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    output.with_suffix(".md").write_text("# M06 Cross-sectional Dynamics audit\n\n- status: **%s**\n- reason: `%s`\n\nNo prediction artifact is produced by this audit.\n" % (status, reason), encoding="utf-8")
    return result
