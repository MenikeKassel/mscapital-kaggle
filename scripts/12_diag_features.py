# -*- coding: utf-8 -*-
"""诊断特征: NaN/Inf 列"""
import numpy as np
import polars as pl

FEAT = r"D:\mscapital-forecasting\data\processed\train_features.parquet"
tr = pl.read_parquet(FEAT)
all_feats = [c for c in tr.columns if c not in ("sample_id", "month", "target")]
arr = tr.select(all_feats).to_numpy().astype(np.float32)

n_nan = np.isnan(arr).sum()
n_inf = np.isinf(arr).sum()
print(f"总 NaN: {n_nan}, 总 Inf: {n_inf}")

bad = []
for i, c in enumerate(all_feats):
    col = arr[:, i]
    nn = np.isnan(col).sum()
    ni = np.isinf(col).sum()
    if nn > 0 or ni > 0:
        bad.append((c, nn, ni, float(np.nanmin(col)), float(np.nanmax(col))))
print(f"问题列数: {len(bad)}")
for c, nn, ni, mn, mx in bad[:30]:
    print(f"  {c}: NaN={nn} Inf={ni} range=[{mn:.4g},{mx:.4g}]")

# 常数列 (std=0)
sd = arr.std(axis=0)
const = [all_feats[i] for i in range(len(all_feats)) if sd[i] == 0]
print(f"\n常数列: {const}")

# 极端值检查 (|x| > 1e6)
n_ext = (np.abs(arr) > 1e6).sum()
print(f"|x|>1e6 的单元数: {n_ext}")
