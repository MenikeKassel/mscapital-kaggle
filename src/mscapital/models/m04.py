"""M04 Optiver Interaction Family under the frozen residual protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..artifacts import array_hash
from ..features.optiver_interactions import EPSILON, OptiverInteractionFrame
from ..residual import CanonicalOOF
from .m01a import M01AConfig
from .m02 import run_m02_outer, summarize_m02


_BUILDER_PROVENANCE = {
    "feature_count": 24,
    "epsilon": EPSILON,
    "signed_flow_transform": "asinh",
    "positive_transform": "log1p",
    "relative_spread_proxy": "max(lob_bid_l1_l2_gap + lob_ask_l1_l2_gap, 0)",
    "input_alignment": "exact sample_id/month/target",
}


def run_m04_outer(
    canonical: CanonicalOOF,
    features: OptiverInteractionFrame,
    baseline_root: str | Path,
    output_root: str | Path,
    outer: str,
    *,
    config: M01AConfig = M01AConfig(),
) -> dict[str, Any]:
    """Train one M04 fold with the unchanged M01-A CatBoost/alpha procedure."""

    result = run_m02_outer(
        canonical,
        features,
        baseline_root,
        output_root,
        outer,
        config=config,
        method_id="m04-optiver-interactions",
        output_subdir="m04-optiver-interactions",
        split_label="m04-optiver-interactions",
        report_label="M04 Optiver Interaction Family",
        include_reference_diagnostics=False,
    )
    output = Path(output_root) / "m04-optiver-interactions" / outer
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = {
        **_BUILDER_PROVENANCE,
        "interaction_values_hash": array_hash(features.values),
    }
    manifest_diagnostics = manifest.setdefault("diagnostics", {})
    manifest_diagnostics["feature_builder"] = provenance
    fingerprints = manifest.setdefault("data_fingerprints", {})
    fingerprints["interaction_values"] = fingerprints.pop("geometry_values")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    report = output / "report.md"
    report.write_text(
        report.read_text(encoding="utf-8")
        + "\n- feature family: `24 fixed Optiver-style interactions`"
        + "\n- relative spread: `normalized M02 geometry proxy (not a raw official column)`\n",
        encoding="utf-8",
    )
    return result | {"feature_builder": provenance}


def summarize_m04(artifact_root: str | Path) -> dict[str, Any]:
    """Replay and summarize all four M04 outer-fold artifacts."""

    result = summarize_m02(
        artifact_root,
        output_subdir="m04-optiver-interactions",
        split_label="m04-optiver-interactions",
        method="M04 Optiver Interaction Family",
    )
    for row in result["rows"]:
        builder = row.get("feature_builder")
        if not isinstance(builder, dict):
            raise ValueError(f"{row['outer']}: M04 feature-builder provenance is missing")
        if any(builder.get(key) != value for key, value in _BUILDER_PROVENANCE.items()):
            raise ValueError(f"{row['outer']}: M04 feature-builder provenance is invalid")
        values_hash = builder.get("interaction_values_hash")
        if not isinstance(values_hash, str) or len(values_hash) != 64:
            raise ValueError(f"{row['outer']}: M04 interaction value hash is invalid")
    return result
