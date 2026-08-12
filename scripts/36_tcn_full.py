# -*- coding: utf-8 -*-
"""
TCN 全量训练 + test 预测 (60ep x 3seed, 全量 0-70月)
输出: p12_full_test_pred.npz (647,896) 供 v6 融合
"""
import os, sys, time
import numpy as np
import polars as pl
import torch

sys.path.insert(0, r"D:\mscapital-kaggle\scripts")
import kaggle_p12_tcn as k

k.DATA = r"D:\mscapital-forecasting\data\raw"
OUT = r"D:\mscapital-forecasting\data\processed\p12_out"
os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}", flush=True)

label = pl.read_ipc(f"{k.DATA}/train/label.feather", memory_map=False)
tr_ids = label["sample_id"].to_numpy()
y_tr = label["target"].to_numpy().astype(np.float32)
te_feat = pl.read_parquet(r"D:\mscapital-forecasting\data\processed\test_features.parquet")
te_ids = te_feat["sample_id"].to_numpy()
print(f"train {len(tr_ids):,} test {len(te_ids):,}", flush=True)

t0 = time.time()
Xf_tr = k.build_fast_tensor("train", tr_ids)
Xs_tr = k.build_slow_tensor("train", tr_ids)
Xf_te = k.build_fast_tensor("test", te_ids)
Xs_te = k.build_slow_tensor("test", te_ids)
print(f"tensors built ({time.time()-t0:.0f}s)", flush=True)

for arr in (Xf_tr, Xf_te):
    mu = Xf_tr.mean(axis=(0, 2), keepdims=True)
    sd = Xf_tr.std(axis=(0, 2), keepdims=True) + 1e-6
    arr[:] = ((arr - mu) / sd).clip(-10, 10)
for arr in (Xs_tr, Xs_te):
    mu = Xs_tr.mean(axis=(0, 2), keepdims=True)
    sd = Xs_tr.std(axis=(0, 2), keepdims=True) + 1e-6
    arr[:] = ((arr - mu) / sd).clip(-10, 10)
Xf_tr = np.nan_to_num(Xf_tr, nan=0.0); Xf_te = np.nan_to_num(Xf_te, nan=0.0)
Xs_tr = np.nan_to_num(Xs_tr, nan=0.0); Xs_te = np.nan_to_num(Xs_te, nan=0.0)

Xft = torch.from_numpy(Xf_tr); Xst = torch.from_numpy(Xs_tr)
Xfte = torch.from_numpy(Xf_te); Xste = torch.from_numpy(Xs_te)
y_mean, y_std = y_tr.mean(), y_tr.std() + 1e-6
y_n = (y_tr - y_mean) / y_std
n = len(y_tr)
EPOCHS, BATCH = 60, 512

def predict_te(model):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xfte), 8192):
            outs.append(model(Xfte[i:i+8192].to(DEVICE), Xste[i:i+8192].to(DEVICE)).cpu().numpy())
    return np.concatenate(outs) * y_std + y_mean

preds = []
T0 = time.time()
for seed in [2026, 7, 123]:
    torch.manual_seed(seed); np.random.seed(seed)
    model = k.DualTower().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = torch.nn.MSELoss()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            opt.zero_grad()
            loss = lossf(model(Xft[idx].to(DEVICE), Xst[idx].to(DEVICE)),
                         torch.from_numpy(y_n[idx]).to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if (ep + 1) % 15 == 0 or ep == EPOCHS:
            print(f"seed{seed} ep{ep+1}: loss={tot/n:.6f} ({time.time()-T0:.0f}s)", flush=True)
    preds.append(predict_te(model))
    print(f"seed{seed} done ({time.time()-T0:.0f}s)", flush=True)

p_ens = np.mean(preds, axis=0)
np.savez(f"{OUT}/p12_full_test_pred.npz", pred=p_ens, test_ids=te_ids)
print(f"\nsaved {OUT}/p12_full_test_pred.npz ({len(p_ens):,}) mean={p_ens.mean():.6f} std={p_ens.std():.6f}")
print(f"TOTAL {time.time()-T0:.0f}s")
