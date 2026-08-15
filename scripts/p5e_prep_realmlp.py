# -*- coding: utf-8 -*-
"""P5-E prep: RealMLP spot-check feature parquets (152 vs 152+Z), months 0-70.

Z 按 R61_70 split 严格 OOF:
  months 0-50  (inner_train + refit): fold-within-0-50 (10 月块, 其他块拟合)
  months 51-60 (inner_tune):           fit on 0-50
  months 61-70 (outer_valid):          fit on 0-60
输出: output/p5b_scfi/spot_rmlp_A.parquet / spot_rmlp_C.parquet (sample_id + 特征)
"""
import gc
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, r"D:\mscapital-kaggle\scripts")
sys.path.insert(0, r"D:\mscapital-kaggle\src")

import p5b_scfi as P

OUT = Path(r"D:\mscapital-kaggle\output\p5b_scfi")
t0 = time.time()

data = P.load_data()
print(f"loaded ({time.time()-t0:.0f}s)", flush=True)
m = data["month"]

# --- Z for 0-50 + 51-60 (nuisance folds within 0-50, eval fit on 0-50) ---
Ztr_a, Ze_a, R2a, (o_cols, t_cols), (madO_a, madT_a) = P.nuisance_crossfit(
    data, np.arange(0, 51), np.arange(51, 61))
nZ = Ztr_a.shape[1]
Z = np.full((len(m), nZ), np.nan)
mask_a = np.isin(m, np.arange(0, 51))
Z[mask_a] = Ztr_a
Z[np.isin(m, np.arange(51, 61))] = Ze_a
del Ztr_a, Ze_a
gc.collect()
print(f"Z 0-60 done ({time.time()-t0:.0f}s)", flush=True)

# --- Z for 61-70 (fit on 0-60) ---
Ztr_b, Ze_b, R2b, _, _ = P.nuisance_crossfit(data, np.arange(0, 61), np.arange(61, 71))
Z[np.isin(m, np.arange(61, 71))] = Ze_b
del Ztr_b, Ze_b
gc.collect()
assert np.isfinite(Z).all(), "Z has non-finite"
print(f"Z 61-70 done ({time.time()-t0:.0f}s)", flush=True)

# --- feature frames ---
fe152 = data["fe152"]
znames = [f"Z_{c}" for c in data["feraw"]]
df = pl.DataFrame({
    "sample_id": data["sample_id"],
    **{c: data["X152"][:, i] for i, c in enumerate(fe152)},
    **{zn: Z[:, j] for j, zn in enumerate(znames)},
})
df_A = df.select(["sample_id"] + fe152)
df_C = df.select(["sample_id"] + fe152 + znames)
df_A.write_parquet(OUT / "spot_rmlp_A.parquet")
df_C.write_parquet(OUT / "spot_rmlp_C.parquet")
print(f"saved A({len(fe152)}f) / C({len(fe152)+nZ}f) ({time.time()-t0:.0f}s) → {OUT}")
