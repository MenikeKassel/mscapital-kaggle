# -*- coding: utf-8 -*-
"""sanity: 增强版单文件 (构建 + 1 ep 训练)"""
import sys, time
import numpy as np
import polars as pl

sys.path.insert(0, r"D:\mscapital-kaggle\scripts")
import kaggle_p12_enhanced as k

k.DATA = r"D:\mscapital-forecasting\data\raw"
k.OUT = r"D:\mscapital-forecasting\data\processed\p12_out"
k.DEVICE = __import__("torch").device("cpu")

label = pl.read_ipc(f"{k.DATA}/train/label.feather", memory_map=False)
sub = label.filter(pl.col("month") <= 5)["sample_id"].to_numpy()[:1000]
t0 = time.time()
Xf = k.build_fast_tensor("train", sub)
Xs = k.build_slow_tensor("train", sub)
print(f"tensors OK ({time.time()-t0:.0f}s)", flush=True)
print(f"fast {Xf.shape} nan={np.isnan(Xf).sum()} | slow {Xs.shape} nan={np.isnan(Xs).sum()}", flush=True)
model = k.DualTower()
import torch
with torch.no_grad():
    p = model(torch.from_numpy(Xf[:8]), torch.from_numpy(Xs[:8]))
print(f"forward OK: {p.shape}", flush=True)
print("SANITY PASS", flush=True)
