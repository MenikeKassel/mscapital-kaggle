# -*- coding: utf-8 -*-
"""验证 m05 market_state 特征 = f0726 列子集 (值级比对)."""
import numpy as np
import polars as pl

ms = pl.read_parquet(r"D:\mscapital-kaggle\output\m05_state\market_state_train.parquet").sort("sample_id")
fe = pl.read_parquet(r"D:\mscapital-forecasting\data\processed\f0726_train_f32.parquet").sort("sample_id")

ms_feats = [c for c in ms.columns if c not in ("sample_id", "month", "target")]
print("m05 features:", ms_feats)

fe_cols = set(fe.columns)
missing = [c for c in ms_feats if c not in fe_cols]
print("m05 特征不在 f0726 中:", missing or "无 — 全部是 f0726 列!")

# 值级比对 (前 200k 行)
m = ms.head(200_000).select(ms_feats).to_numpy()
f = fe.head(200_000).select(ms_feats).to_numpy()
same = np.isclose(m, f, rtol=1e-5, atol=1e-8, equal_nan=True).mean()
print(f"值级一致率 (200k 行): {same:.6f}")
# 不一致的列
bad = []
for i, c in enumerate(ms_feats):
    ok = np.isclose(m[:, i], f[:, i], rtol=1e-5, atol=1e-8, equal_nan=True).mean()
    if ok < 0.9999:
        bad.append((c, round(float(ok), 5)))
print("不一致列:", bad or "全部一致")
