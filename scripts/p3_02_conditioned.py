# -*- coding: utf-8 -*-
"""P3-02: Market-State Conditioning v2.

Residual model input = [11 ReVol-lite state features] + [24 M01-A event flow],
i.e. event flow conditioned on market state. Runs the frozen M01-A protocol
(fit -> inner-tune alpha -> refit -> outer) for PSEUDO/H2/T3/T4 and applies
the protocol-v2 gate.

Baselines for comparison:
- E01 ReVol-lite only (11 state): PSEUDO +0.00110, gate False
- M01-A event flow only (24):     PSEUDO +0.00000, gate False
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from mscapital.models.m01a import (
    fit_m01a_selection,
    run_m01a_outer,
    summarize_m01a,
)
from mscapital.models.revol_lite import load_revol_lite_frame
from mscapital.models.m01a import load_event_flow_frame
from mscapital.residual import CanonicalOOF

from p3_common import concat_frames, load_p3_frame

CANONICAL = Path(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
REVOL = Path(r"D:\mscapital-kaggle\output\e01_revol_lite_features\revol_lite_train.parquet")
EVENT_FLOW = Path(r"D:\mscapital-kaggle\output\m01a_features\event_flow_train.parquet")
BASELINE_ROOT = Path(r"D:\mscapital-kaggle\output\c4_protocol_closed_final\clean-baseline-v2")
OUT = Path(r"D:\mscapital-kaggle\output\p3_02_conditioned_formal")


def main() -> None:
    canonical = CanonicalOOF(**{
        k: np.asarray(np.load(CANONICAL)[k]) for k in
        ("sample_id", "month", "target", "baseline_oof", "source_train_end")
    })
    canonical.validate()

    revol = load_revol_lite_frame(REVOL)
    flow = load_event_flow_frame(EVENT_FLOW)
    combined = concat_frames(revol, flow)
    print(f"combined features: {len(combined.feature_names)} rows={combined.sample_id.size}")

    results = []
    for outer in ("PSEUDO", "H2", "T3", "T4"):
        diag = run_m01a_outer(
            canonical, combined, BASELINE_ROOT, OUT, outer,
        )
        results.append((outer, diag["delta_vs_baseline"], diag["final_score"]))
        print(f"{outer}: delta={diag['delta_vs_baseline']:+.9f} score={diag['final_score']:.9f}")

    summary = summarize_m01a(OUT)
    print("\n=== P3-02 gate ===")
    print(f"PSEUDO delta={summary['rows'][0]['delta_vs_baseline']:+.9f}")
    print(f"mean delta={summary['mean_delta']:+.9f}")
    print(f"gate={summary['gate']}")


if __name__ == "__main__":
    main()
