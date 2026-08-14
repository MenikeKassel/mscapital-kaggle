# -*- coding: utf-8 -*-
"""Smoke test for p5_02i_info_audit: transforms + tiny end-to-end training."""
import sys
sys.path.insert(0, r"D:\mscapital-kaggle\scripts")
import numpy as np
import torch
import p5_02i_info_audit as A

# 1) surrogate transforms on fake data (shape / finiteness / phase variance)
rng = np.random.default_rng(0)
raw = rng.normal(size=(3000, 200, 18)).astype(np.float32)
for arm in ["shuffle", "reverse", "block10", "block20", "block50", "desync", "phase"]:
    X = A.transform(raw, arm, seed=99)
    assert X.shape == raw.shape, arm
    assert np.isfinite(X).all(), arm
    if arm == "phase":
        print("phase var ratio (should be ~1):", float((X.var(axis=1) / raw.var(axis=1)).mean()))
    if arm == "shuffle":
        # marginal distribution preserved
        print("shuffle mean/std preserved:", float(X.mean()), float(X.std()), float(raw.mean()), float(raw.std()))
print("transforms OK")

# 2) tiny end-to-end: real data, small subset
lab = A.pl.read_ipc(A.LABEL)
m = lab["month"].to_numpy()
sid = lab["sample_id"].to_numpy()
y = lab["target"].to_numpy()
sel = np.where((m >= 21) & (m <= 40))[0][:1500]
selv = np.where((m >= 51) & (m <= 70))[0][:400]
Xtr = A.build_sequence(A.MARKET, sid[sel])
Xva = A.build_sequence(A.MARKET, sid[selv])
print("built", Xtr.shape, Xva.shape)
mu = Xtr.reshape(-1, A.CHANNELS).mean(0, keepdims=True).astype(np.float32)
sd = Xtr.reshape(-1, A.CHANNELS).std(0, keepdims=True).astype(np.float32) + 1e-6
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for arm in ["raw", "shuffle", "desync", "phase"]:
    Xa = Xtr if arm == "raw" else A.transform(Xtr, arm, 123)
    Xv = Xva if arm == "raw" else A.transform(Xva, arm, 123)
    mdl = A.train_arm((Xa - mu) / sd, y[sel].astype(np.float32), dev, seed=7)
    p = A.predict(mdl, (Xv - mu) / sd, dev)
    print(arm, "corr(y) =", f"{np.corrcoef(p, y[selv])[0, 1]:+.4f}")
    del Xa, Xv, mdl
mdl = A.train_arm((Xtr - mu) / sd, (y[sel] > 0).astype(np.float32), dev, seed=7, head="cls")
p = A.predict(mdl, (Xva - mu) / sd, dev)
print("cls probe auc =", f"{A.auc((y[selv] > 0).astype(float), p):.4f}")
print("SMOKE OK")
