# -*- coding: utf-8 -*-
"""
RealMLP 30ep 全量生产推理 (云端): Clean Baseline v2 组件
train 0-70 全量 → test 预测 → 保存 npz
"""
# P100 (sm_60) 兼容: torch 2.2.2 + numpy<2 (已验证方案)
import subprocess, sys, os
try:
    import torch as _t0
    _cap = _t0.cuda.get_device_capability(0) if _t0.cuda.is_available() else (0, 0)
except Exception:
    _cap = (0, 0)
if _cap[0] and _cap[0] < 7:
    print(f"GPU capability {_cap} < 7.0, downgrading torch...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch==2.2.2", "numpy<2"])
    os.execv(sys.executable, [sys.executable] + sys.argv)
print("torch OK", flush=True)

sys.path.insert(0, "/kaggle/input/msc-src-pkg")
import time
import numpy as np
import pandas as pd

t0 = time.time()
from mscapital.models.realmlp import (
    RealMLPConfig, load_frame, CleanRealMLPPreprocessor, _train_refit_predict,
)

DATA = "/kaggle/input/msc-f0726-pq"
LABEL = "/kaggle/input/competitions/ms-capital-real-financial-market-forecasting/train/label.feather"
OUT = "/kaggle/working"

cfg = RealMLPConfig(epochs=30)
print("cfg epochs:", cfg.epochs, flush=True)

# 全量 train
frame = load_frame(f"{DATA}/f0726_train_f32.parquet", LABEL)
print(f"train frame: {len(frame.sample_id):,} ({time.time()-t0:.0f}s)", flush=True)

pre = CleanRealMLPPreprocessor(feature_names=tuple(frame.features.columns), config=cfg)
pre.fit(frame.features, frame.target)
x, c = pre.transform(frame.features)
print(f"preprocessed: {x.shape} cats={c.shape[1]} selected={len(pre.selected_numeric)} ({time.time()-t0:.0f}s)", flush=True)

# test
test = pd.read_parquet(f"{DATA}/f0726_test_f32.parquet").sort_values("sample_id").reset_index(drop=True)
x_te, c_te = pre.transform(test[list(frame.features.columns)])
print(f"test preprocessed: {x_te.shape} ({time.time()-t0:.0f}s)", flush=True)

# 训练 30ep 全量 + test 预测 (EMA best)
pred = _train_refit_predict(x, c, frame.target, x_te, c_te, 1.0, cfg)
np.savez(f"{OUT}/realmlp_prod_test_pred.npz", pred=pred, test_ids=test["sample_id"].to_numpy())
print(f"saved realmlp_prod_test_pred.npz ({len(pred):,})", flush=True)
print(f"mean={pred.mean():.2e} std={pred.std():.2e}", flush=True)
print("ALL DONE", flush=True)
