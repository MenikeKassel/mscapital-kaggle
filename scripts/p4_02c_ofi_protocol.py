# -*- coding: utf-8 -*-
"""P4-02C: low-overlap LB142 market-form factors -> frozen M01-A protocol.

Frame = m_ofi_sum (9 windows) + m_imb_mean (9) + m_imb2_mean_600 + short-window
mid_std/mid_last (5/15/30) = 24 low-overlap features.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from mscapital.models.m01a import run_m01a_outer, summarize_m01a
from mscapital.residual import CanonicalOOF

from p3_common import save_p3_features, load_p3_frame

CANONICAL = Path(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
BASELINE_ROOT = Path(r"D:\mscapital-kaggle\output\c4_protocol_closed_final\clean-baseline-v2")
NPZ = Path(r"D:\mscapital-kaggle\output\p4_02b_market_forms\market_forms.npz")
OUT = Path(r"D:\mscapital-kaggle\output\p4_02c_ofi_protocol")
FEATURE_OUT = Path(r"D:\mscapital-kaggle\output\p4_02c_features")
OUT.mkdir(parents=True, exist_ok=True)
FEATURE_OUT.mkdir(parents=True, exist_ok=True)

PICK = [  # low-overlap forms
    "m_ofi_sum_5", "m_ofi_sum_15", "m_ofi_sum_30", "m_ofi_sum_45", "m_ofi_sum_60",
    "m_ofi_sum_120", "m_ofi_sum_180", "m_ofi_sum_240", "m_ofi_sum_300", "m_ofi_sum_600",
    "m_imb_mean_5", "m_imb_mean_15", "m_imb_mean_30", "m_imb_mean_45", "m_imb_mean_60",
    "m_imb_mean_120", "m_imb_mean_180", "m_imb_mean_240", "m_imb_mean_300", "m_imb_mean_600",
    "m_imb2_mean_600", "m_mid_std_5", "m_mid_std_15", "m_mid_last_5",
]


def main() -> None:
    canonical = CanonicalOOF(**{
        k: np.asarray(np.load(CANONICAL)[k]) for k in
        ("sample_id", "month", "target", "baseline_oof", "source_train_end")
    })
    canonical.validate()
    d = np.load(NPZ)
    X_all, names = d["X"], [str(x) for x in d["names"]]
    idx = [names.index(p) for p in PICK]
    vals = X_all[:, idx]
    names_pick = tuple(PICK)

    fp = FEATURE_OUT / "ofi_features.parquet"
    save_p3_features(fp, "p4-02c-ofi", names_pick,
                     canonical.sample_id, canonical.month, canonical.target, vals)
    frame = load_p3_frame(fp, names_pick)
    for outer in ("PSEUDO", "H2", "T3", "T4"):
        diag = run_m01a_outer(canonical, frame, BASELINE_ROOT, OUT, outer)
        print(f"{outer}: delta={diag['delta_vs_baseline']:+.9f} alpha={diag['alpha']:.2f} "
              f"score={diag['final_score']:.9f}")
    summ = summarize_m01a(OUT)
    print("\n=== P4-02C gate ===")
    for row in summ["rows"]:
        print(f"  {row['outer']}: delta={row['delta_vs_baseline']:+.9f}")
    print(f"mean_delta={summ['mean_delta']:+.9f} gate={summ['gate']}")


if __name__ == "__main__":
    main()
