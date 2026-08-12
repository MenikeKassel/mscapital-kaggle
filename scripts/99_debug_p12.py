# -*- coding: utf-8 -*-
"""诊断: tensor 中的 NaN/inf 分布"""
import sys, time
import numpy as np
import polars as pl

sys.path.insert(0, r"D:\mscapital-kaggle\scripts")
import kaggle_p12_tcn as k

k.DATA = r"D:\mscapital-forecasting\data\raw"
label = pl.read_ipc(f"{k.DATA}/train/label.feather", memory_map=False)
sub = label.filter(pl.col("month") <= 32)["sample_id"].to_numpy()[:5000]

Xf = k.build_fast_tensor("train", sub)
Xs = k.build_slow_tensor("train", sub)

for name, X in [("fast", Xf), ("slow", Xs)]:
    nan_n = np.isnan(X).sum()
    inf_n = np.isinf(X).sum()
    mx = np.nanmax(np.abs(X)) if nan_n < X.size else float("nan")
    print(f"{name}: shape={X.shape} nan={nan_n} inf={inf_n} max_abs={mx}")
    if nan_n or inf_n:
        # 定位通道
        for ch in range(X.shape[1]):
            c_nan = np.isnan(X[:, ch]).sum()
            c_inf = np.isinf(X[:, ch]).sum()
            if c_nan or c_inf:
                print(f"  ch{ch}: nan={c_nan} inf={c_inf}")
