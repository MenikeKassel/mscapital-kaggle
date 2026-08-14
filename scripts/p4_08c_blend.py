# -*- coding: utf-8 -*-
"""P4-08C: blend cosine-RealMLP test pred with v7 -> submissions.

OOF evidence: final = base + alpha * unit(cosine_pred), alpha 0.10-0.25
(unc-version mean delta +0.009 at alpha 0.12-0.13).
Production has no target -> generate several alpha versions, unit-mixed and
rescaled to v7 scale for submission format consistency.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

P12 = Path(r"D:\mscapital-forecasting\data\processed\p12_out")
V7 = Path(r"D:\mscapital-kaggle\output\submissions\submission_blend_v7_rl20.csv")
OUT = Path(r"D:\mscapital-kaggle\output\submissions")
OUT.mkdir(parents=True, exist_ok=True)


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    return x / (np.linalg.norm(x) + 1e-12)


def main() -> None:
    cos = np.load(P12 / "realmlp_cosine_test_pred.npz")
    p_cos = cos["pred"].astype(np.float64)
    ids = cos["test_ids"].astype(np.int64)
    v7 = pl.read_csv(V7).sort("sample_id")
    assert np.array_equal(v7["sample_id"].to_numpy(), ids), "id mismatch"
    p_v7 = v7["prediction"].to_numpy().astype(np.float64)
    print(f"corr(cosine, v7) = {np.corrcoef(p_cos, p_v7)[0, 1]:.4f}")
    print(f"cosine: mean={p_cos.mean():.2e} std={p_cos.std():.2e}")
    print(f"v7    : mean={p_v7.mean():.2e} std={p_v7.std():.2e}")

    u_v7, u_cos = unit(p_v7), unit(p_cos)
    for a in (0.13, 0.17, 0.10, 0.25):  # 0.13 = uncentered-OOF 最优 (首选)
        mix = u_v7 + a * u_cos
        pred = mix / mix.std() * p_v7.std() + p_v7.mean()  # rescale to v7 scale
        sub = pl.DataFrame({"sample_id": pl.Series(ids, dtype=pl.Int32),
                            "prediction": pl.Series(pred, dtype=pl.Float64)}).sort("sample_id")
        fp = OUT / f"submission_v9_cos_a{int(a * 100)}.csv"
        sub.write_csv(fp)
        print(f"w={a:.2f}: saved {fp.name} (mean={pred.mean():.2e} std={pred.std():.2e})")
    # cosine solo (rescaled to v7 scale)
    pred = u_cos / u_cos.std() * p_v7.std() + p_v7.mean()
    pl.DataFrame({"sample_id": pl.Series(ids, dtype=pl.Int32),
                  "prediction": pl.Series(pred, dtype=pl.Float64)}).sort("sample_id") \
        .write_csv(OUT / "submission_v9_cos_solo.csv")
    print("saved submission_v9_cos_solo.csv")


if __name__ == "__main__":
    main()
