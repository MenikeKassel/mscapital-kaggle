# -*- coding: utf-8 -*-
"""
P1-2d: TCN 增强版 (Kaggle GPU) — 自包含单文件
FAST tower (order+tx 60s x 16ch) + SLOW tower (market 60bar x 8ch)
每 fold: 构建 tensor → 60ep x 3seed 集成 → best-state 保存 pred
folds: PSEUDO (3seed 主) / T3 / T4 (1seed 验证)
"""
import os, time
import numpy as np
import polars as pl
import torch
import torch.nn as nn

DATA = "/kaggle/input/competitions/ms-capital-real-financial-market-forecasting"
OUT = "/kaggle/working"

# Kaggle 可能分配 Tesla P100 (sm_60), 预装 torch 不支持 → 降级 torch 2.2.2 (支持 sm_50+)
import subprocess, sys
try:
    import torch
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        if cap[0] < 7:
            print(f"GPU capability {cap} < 7.0, downgrading torch...", flush=True)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch==2.2.2", "numpy<2"])
            os.execv(sys.executable, [sys.executable] + sys.argv)
except Exception as e:
    print(f"torch check failed: {e}", flush=True)

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}", flush=True)

def build_fast_tensor(split, sample_ids):
    t0 = time.time()
    ord_lf = (pl.scan_ipc(f"{DATA}/{split}/order.feather", memory_map=False)
              .filter(pl.col("sample_id").is_in(sample_ids))
              .with_columns(pl.col("seconds_before_predict").round(0).cast(pl.Int32).alias("sec"))
              .with_columns([
                  pl.when(pl.col("side") == 0).then(1.0).otherwise(0.0).alias("_buy"),
                  pl.when(pl.col("side") == 1).then(1.0).otherwise(0.0).alias("_sell"),
                  pl.when(pl.col("order_action") == 0).then(1.0).otherwise(0.0).alias("_new"),
                  pl.when(pl.col("order_action") == 1).then(1.0).otherwise(0.0).alias("_can"),
              ]))
    o = (ord_lf.group_by(["sample_id", "sec"]).agg([
        (pl.col("_buy") * pl.col("volume")).sum().alias("ch_buy_ord"),
        (pl.col("_sell") * pl.col("volume")).sum().alias("ch_sell_ord"),
        (pl.col("_buy") * pl.col("_new") * pl.col("volume")).sum().alias("ch_add_buy"),
        (pl.col("_sell") * pl.col("_new") * pl.col("volume")).sum().alias("ch_add_sell"),
        (pl.col("_buy") * pl.col("_can") * pl.col("volume")).sum().alias("ch_can_buy"),
        (pl.col("_sell") * pl.col("_can") * pl.col("volume")).sum().alias("ch_can_sell"),
        pl.col("volume").sum().alias("ch_ord_vol"),
        pl.len().alias("ch_ord_cnt"),
    ]).collect(streaming=True))
    tx_lf = (pl.scan_ipc(f"{DATA}/{split}/transaction.feather", memory_map=False)
             .filter(pl.col("sample_id").is_in(sample_ids))
             .with_columns(pl.col("seconds_before_predict").round(0).cast(pl.Int32).alias("sec"))
             .with_columns([
                 pl.when(pl.col("side") == 0).then(1.0).otherwise(0.0).alias("_buy"),
                 pl.when(pl.col("side") == 1).then(1.0).otherwise(0.0).alias("_sell"),
             ]))
    t = (tx_lf.group_by(["sample_id", "sec"]).agg([
        (pl.col("_buy") * pl.col("volume")).sum().alias("ch_buy_tx"),
        (pl.col("_sell") * pl.col("volume")).sum().alias("ch_sell_tx"),
        pl.col("volume").sum().alias("ch_tx_vol"),
        pl.len().alias("ch_tx_cnt"),
    ]).collect(streaming=True))
    df = o.join(t, on=["sample_id", "sec"], how="full", coalesce=True)
    df = df.with_columns([pl.col(c).fill_null(0.0) for c in df.columns if c not in ("sample_id", "sec")])
    df = df.sort(["sample_id", "sec"])
    n = len(sample_ids)
    T = 60
    tensor = np.zeros((n, 16, T), dtype=np.float32)
    sid_idx = {s: i for i, s in enumerate(sample_ids)}
    ch_map = {
        "ch_buy_ord": 0, "ch_sell_ord": 1, "ch_add_buy": 2, "ch_add_sell": 3,
        "ch_can_buy": 4, "ch_can_sell": 5, "ch_ord_vol": 6, "ch_ord_cnt": 7,
        "ch_buy_tx": 8, "ch_sell_tx": 9, "ch_tx_vol": 10, "ch_tx_cnt": 11,
    }
    ids = df["sample_id"].to_numpy()
    secs = df["sec"].to_numpy().astype(np.int32)
    rows = np.array([sid_idx.get(s, -1) for s in ids])
    ok = (rows >= 0) & (secs >= 1) & (secs <= T)
    rows, secs = rows[ok], secs[ok]
    tcol = T - secs
    for name, c in ch_map.items():
        vals = df[name].to_numpy().astype(np.float32)
        tensor[rows, c, tcol] = vals[ok]
    eps = 1e-6
    tot_ord = tensor[:, 0] + tensor[:, 1] + eps
    tot_tx = tensor[:, 8] + tensor[:, 9] + eps
    tensor[:, 12] = (tensor[:, 0] - tensor[:, 1]) / tot_ord
    tensor[:, 13] = (tensor[:, 8] - tensor[:, 9]) / tot_tx
    tensor[:, 14] = (tensor[:, 2] + tensor[:, 3]) / (tensor[:, 4] + tensor[:, 5] + 1.0)
    tensor[:, 15] = np.log1p(tensor[:, 10])
    tensor = np.nan_to_num(tensor, nan=0.0, posinf=50.0, neginf=-50.0)
    tensor = np.clip(tensor, -50.0, 50.0)
    print(f"fast tensor: {tensor.shape} ({time.time()-t0:.0f}s)", flush=True)
    return tensor

def build_slow_tensor(split, sample_ids):
    t0 = time.time()
    lf = (pl.scan_ipc(f"{DATA}/{split}/market.feather", memory_map=False)
          .filter(pl.col("sample_id").is_in(sample_ids))
          .with_columns((pl.col("seconds_before_predict") / 10.0).round(0).cast(pl.Int32).alias("bar"))
          .with_columns([
              (((pl.col("ask_price_1") + pl.col("bid_price_1")) * 0.5).cast(pl.Float64)).alias("_mid"),
              (((pl.col("ask_price_1") - pl.col("bid_price_1")) / ((pl.col("ask_price_1") + pl.col("bid_price_1")) * 0.5 + 1e-8)).cast(pl.Float64)).alias("_sp_rel"),
              (((pl.col("ask_volume_1") - pl.col("bid_volume_1")) / (pl.col("ask_volume_1") + pl.col("bid_volume_1") + 1.0)).cast(pl.Float64)).alias("_imb1"),
              (((pl.col("ask_volume_2") - pl.col("bid_volume_2")) / (pl.col("ask_volume_2") + pl.col("bid_volume_2") + 1.0)).cast(pl.Float64)).alias("_imb2"),
              ((pl.col("ask_volume_1") + pl.col("bid_volume_1") + pl.col("ask_volume_2") + pl.col("bid_volume_2")).cast(pl.Float64)).alias("_depth"),
          ])
          .with_columns(pl.col("_mid").pct_change().over("sample_id").fill_null(0.0).cast(pl.Float64).alias("_ret")))
    agg = (lf.group_by(["sample_id", "bar"]).agg([
        pl.col("_mid").last().alias("mid_last"),
        pl.col("_sp_rel").mean().cast(pl.Float64).alias("sp_rel"),
        pl.col("_imb1").mean().cast(pl.Float64).alias("imb1"),
        pl.col("_imb2").mean().cast(pl.Float64).alias("imb2"),
        pl.col("_depth").mean().log1p().cast(pl.Float64).alias("depth_log"),
        pl.col("_ret").std().fill_null(0.0).cast(pl.Float64).alias("rv"),
        pl.col("transaction_volume").sum().log1p().cast(pl.Float64).alias("txv_log"),
        pl.col("transaction_count").sum().log1p().cast(pl.Float64).alias("txc_log"),
    ]).sort(["sample_id", "bar"]).collect(streaming=True))
    n = len(sample_ids)
    S = 60
    tensor = np.zeros((n, 8, S), dtype=np.float32)
    sid_idx = {s: i for i, s in enumerate(sample_ids)}
    ids = agg["sample_id"].to_numpy()
    bars = agg["bar"].to_numpy().astype(np.int32)
    rows = np.array([sid_idx.get(s, -1) for s in ids])
    ok = (rows >= 0) & (bars >= 1) & (bars <= S)
    rows, bars = rows[ok], bars[ok]
    tcol = S - bars
    for ch, name in enumerate(["sp_rel", "imb1", "imb2", "depth_log", "rv", "txv_log", "txc_log"]):
        vals = agg[name].to_numpy().astype(np.float32)
        tensor[rows, ch, tcol] = vals[ok]
    ml = agg["mid_last"].to_numpy().astype(np.float64)
    prev = np.zeros(len(agg), dtype=np.float64)
    is_first = np.concatenate([[True], ids[1:] != ids[:-1]])
    prev[1:] = ml[:-1]
    prev[is_first] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where((prev > 0) & (~is_first), (ml - prev) / prev, 0.0)
    tensor[rows, 7, tcol] = ret[ok]
    print(f"slow tensor: {tensor.shape} ({time.time()-t0:.0f}s)", flush=True)
    return tensor

class ResTCNBlock(nn.Module):
    def __init__(self, cin, cout, dilation, p=0.1):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv1d(cout, cout, 3, padding=dilation, dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        self.norm = nn.BatchNorm1d(cout)
    def forward(self, x):
        r = self.skip(x)
        x = self.act(self.c1(x)); x = self.drop(x)
        x = self.act(self.c2(x)); x = self.drop(x)
        return self.norm(x + r)

class SmallTCN(nn.Module):
    def __init__(self, n_ch, out_dim=32, h=48):
        super().__init__()
        self.blocks = nn.Sequential(
            ResTCNBlock(n_ch, h, 1), ResTCNBlock(h, h, 2),
            ResTCNBlock(h, h, 4), ResTCNBlock(h, h, 8))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(h, out_dim))
    def forward(self, x):
        return self.head(self.blocks(x))

class DualTower(nn.Module):
    def __init__(self, n_fast=16, n_slow=8):
        super().__init__()
        self.fast = SmallTCN(n_fast, out_dim=32)
        self.slow = SmallTCN(n_slow, out_dim=24)
        self.head = nn.Sequential(nn.Linear(56, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, xf, xs):
        z = torch.cat([self.fast(xf), self.slow(xs)], dim=1)
        return self.head(z).squeeze(-1)

def cos_uncenter(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def train_fold(label, fname, a, b, c0, d0, n_seeds, epochs=60, batch=512):
    m_all = label["month"].to_numpy().astype(np.int32)
    y_all = label["target"].to_numpy().astype(np.float32)
    sid_all = label["sample_id"].to_numpy()
    sel_t = (m_all >= a) & (m_all <= b)
    sel_v = (m_all >= c0) & (m_all <= d0)
    tr_ids, va_ids = sid_all[sel_t], sid_all[sel_v]
    y_tr, y_va = y_all[sel_t], y_all[sel_v]
    print(f"\n=== {fname}: train m{a}-{b} ({len(tr_ids):,}) valid m{c0}-{d0} ({len(va_ids):,}) ===", flush=True)
    t0 = time.time()
    Xf_tr = build_fast_tensor("train", tr_ids)
    Xs_tr = build_slow_tensor("train", tr_ids)
    Xf_va = build_fast_tensor("train", va_ids)
    Xs_va = build_slow_tensor("train", va_ids)
    print(f"tensors built ({time.time()-t0:.0f}s)", flush=True)
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

    def predict_valid(model):
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
        model = DualTower().to(DEVICE)
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
            pv = predict_valid(model)
            c = cos_uncenter(pv, y_va)
            if c > best_c:
                best_c = c
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if (ep + 1) % 15 == 0 or ep == epochs - 1:
                print(f"  seed{seed} ep{ep+1}: loss={tot/n:.6f} cos={c:.6f} best={best_c:.6f} ({time.time()-T0:.0f}s)", flush=True)
        model.load_state_dict(best_state)
        preds.append(predict_valid(model))
        print(f"  seed{seed} best_cos={best_c:.6f}", flush=True)
    p_ens = np.mean(preds, axis=0)
    c_ens = cos_uncenter(p_ens, y_va)
    np.savez(f"{OUT}/p12_{fname}.npz", pred=p_ens, y=y_va, cos=c_ens)
    print(f"=== {fname}: ens cos={c_ens:.6f} ({time.time()-T0:.0f}s) saved ===", flush=True)

def main():
    label = pl.read_ipc(f"{DATA}/train/label.feather", memory_map=False)
    FOLDS = {"PSEUDO": (0, 32, 33, 70, 3), "T3": (0, 50, 51, 60, 1), "T4": (0, 50, 61, 70, 1)}
    T_ALL = time.time()
    for fname, (a, b, c0, d0, ns) in FOLDS.items():
        train_fold(label, fname, a, b, c0, d0, ns)
    print(f"\nALL DONE ({time.time()-T_ALL:.0f}s)")

if __name__ == "__main__":
    main()
