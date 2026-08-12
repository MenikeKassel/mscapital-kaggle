# -*- coding: utf-8 -*-
"""
P1-2d: TCN 增强 (60ep × 3seed 集成) + 多 fold 验证 (PSEUDO/T3/T4)
每 fold: 构建 tensor → 训练 → best-state 保存 pred
"""
import os, sys, time
import numpy as np
import polars as pl
import torch
import torch.nn as nn

sys.path.insert(0, r"D:\mscapital-kaggle\scripts")
import kaggle_p12_tcn as k

k.DATA = r"D:\mscapital-forecasting\data\raw"
OUT = r"D:\mscapital-forecasting\data\processed\p12_out"
os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}", flush=True)

label = pl.read_ipc(f"{k.DATA}/train/label.feather", memory_map=False)
y_all = label["target"].to_numpy().astype(np.float32)
m_all = label["month"].to_numpy().astype(np.int32)
sid_all = label["sample_id"].to_numpy()

FOLDS = {
    "PSEUDO": (0, 32, 33, 70, 3),   # 主: 3 seeds
    "T3": (0, 50, 51, 60, 1),       # 验证: 1 seed
    "T4": (0, 50, 61, 70, 1),       # 验证: 1 seed
}

def cos_uncenter(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def train_fold(fname, a, b, c0, d0, n_seeds, epochs=60, batch=512):
    sel_t = (m_all >= a) & (m_all <= b)
    sel_v = (m_all >= c0) & (m_all <= d0)
    tr_ids, va_ids = sid_all[sel_t], sid_all[sel_v]
    y_tr, y_va = y_all[sel_t], y_all[sel_v]
    print(f"\n=== {fname}: train m{a}-{b} ({len(tr_ids):,}) valid m{c0}-{d0} ({len(va_ids):,}) ===", flush=True)
    t0 = time.time()
    Xf_tr = k.build_fast_tensor("train", tr_ids)
    Xs_tr = k.build_slow_tensor("train", tr_ids)
    Xf_va = k.build_fast_tensor("train", va_ids)
    Xs_va = k.build_slow_tensor("train", va_ids)
    print(f"tensors built ({time.time()-t0:.0f}s)", flush=True)

    # per-channel 标准化 (train 统计)
    for arr in (Xf_tr, Xf_va):
        mu = Xf_tr.mean(axis=(0, 2), keepdims=True)
        sd = Xf_tr.std(axis=(0, 2), keepdims=True) + 1e-6
        arr[:] = ((arr - mu) / sd).clip(-10, 10)
    for arr in (Xs_tr, Xs_va):
        mu = Xs_tr.mean(axis=(0, 2), keepdims=True)
        sd = Xs_tr.std(axis=(0, 2), keepdims=True) + 1e-6
        arr[:] = ((arr - mu) / sd).clip(-10, 10)
    Xf_tr = np.nan_to_num(Xf_tr, nan=0.0); Xf_va = np.nan_to_num(Xf_va, nan=0.0)
    Xs_tr = np.nan_to_num(Xs_tr, nan=0.0); Xs_va = np.nan_to_num(Xs_va, nan=0.0)

    Xft = torch.from_numpy(Xf_tr); Xst = torch.from_numpy(Xs_tr)
    Xfv = torch.from_numpy(Xf_va); Xsv = torch.from_numpy(Xs_va)
    y_mean, y_std = y_tr.mean(), y_tr.std() + 1e-6
    y_n = (y_tr - y_mean) / y_std
    n = len(y_tr)

    def predict_valid():
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(Xfv), 8192):
                outs.append(model(Xfv[i:i+8192].to(DEVICE), Xsv[i:i+8192].to(DEVICE)).cpu().numpy())
        return np.concatenate(outs) * y_std + y_mean

    preds = []
    T0 = time.time()
    for seed in [2026, 7, 123][:n_seeds]:
        torch.manual_seed(seed); np.random.seed(seed)
        model = k.DualTower().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        lossf = nn.MSELoss()
        best_c, best_state = -1, None
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, batch):
                idx = perm[i:i+batch]
                opt.zero_grad()
                loss = lossf(model(Xft[idx].to(DEVICE), Xst[idx].to(DEVICE)),
                             torch.from_numpy(y_n[idx]).to(DEVICE))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(idx)
            sched.step()
            pv = predict_valid()
            c = cos_uncenter(pv, y_va)
            if c > best_c:
                best_c = c
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if (ep + 1) % 15 == 0 or ep == epochs - 1:
                print(f"  seed{seed} ep{ep+1}: loss={tot/n:.6f} cos={c:.6f} best={best_c:.6f} ({time.time()-T0:.0f}s)", flush=True)
        model.load_state_dict(best_state)
        preds.append(predict_valid())
        print(f"  seed{seed} best_cos={best_c:.6f}", flush=True)

    p_ens = np.mean(preds, axis=0)
    c_ens = cos_uncenter(p_ens, y_va)
    np.savez(f"{OUT}/p12_{fname}.npz", pred=p_ens, y=y_va, cos=c_ens)
    print(f"=== {fname}: ens cos={c_ens:.6f} ({time.time()-T0:.0f}s) saved ===", flush=True)
    return c_ens

T_ALL = time.time()
for fname, (a, b, c0, d0, ns) in FOLDS.items():
    train_fold(fname, a, b, c0, d0, ns)
print(f"\nALL DONE ({time.time()-T_ALL:.0f}s)")
