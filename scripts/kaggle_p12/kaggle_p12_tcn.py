# -*- coding: utf-8 -*-
"""
P1-2: 1s dual-tower TCN (Kaggle GPU / 本地)
FAST tower: order+tx 60s x 1s, 16ch -> TCN
SLOW tower: market 10min x 10s, 8ch -> TCN
协议: PSEUDO-LB38 fold (train m0-32 / valid m33-70)
输出: valid 预测 npz + cos
"""
import os, time, math
import numpy as np
import polars as pl
import torch
import torch.nn as nn

DATA = "/kaggle/input/competitions/ms-capital-real-financial-market-forecasting"
OUT = "/kaggle/working"
SEED = 2026

# ============ 1. 秒级聚合构建 ============
def build_fast_tensor(split, sample_ids):
    """order+tx 60s x 1s x 16ch; 返回 (N, 16, 60) float32"""
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
    tensor[:, 14] = (tensor[:, 2] + tensor[:, 3]) / (tensor[:, 4] + tensor[:, 5] + eps)
    tensor[:, 15] = np.log1p(tensor[:, 10])
    print(f"fast tensor: {tensor.shape} ({time.time()-t0:.0f}s)", flush=True)
    return tensor

def build_slow_tensor(split, sample_ids):
    """market 10min -> 10s bars, 60 steps x 8ch"""
    t0 = time.time()
    lf = (pl.scan_ipc(f"{DATA}/{split}/market.feather", memory_map=False)
          .filter(pl.col("sample_id").is_in(sample_ids))
          .with_columns((pl.col("seconds_before_predict") / 10.0).round(0).cast(pl.Int32).alias("bar"))
          .with_columns([
              ((pl.col("ask_price_1") + pl.col("bid_price_1")) * 0.5).alias("_mid"),
              ((pl.col("ask_price_1") - pl.col("bid_price_1")) / ((pl.col("ask_price_1") + pl.col("bid_price_1")) * 0.5 + 1e-8)).alias("_sp_rel"),
              ((pl.col("ask_volume_1") - pl.col("bid_volume_1")) / (pl.col("ask_volume_1") + pl.col("bid_volume_1") + 1.0)).alias("_imb1"),
              ((pl.col("ask_volume_2") - pl.col("bid_volume_2")) / (pl.col("ask_volume_2") + pl.col("bid_volume_2") + 1.0)).alias("_imb2"),
              (pl.col("ask_volume_1") + pl.col("bid_volume_1") + pl.col("ask_volume_2") + pl.col("bid_volume_2")).alias("_depth"),
          ])
          .with_columns(pl.col("_mid").pct_change().over("sample_id").fill_null(0.0).alias("_ret")))
    agg = (lf.group_by(["sample_id", "bar"]).agg([
        pl.col("_mid").last().alias("mid_last"),
        pl.col("_sp_rel").mean().alias("sp_rel"),
        pl.col("_imb1").mean().alias("imb1"),
        pl.col("_imb2").mean().alias("imb2"),
        pl.col("_depth").mean().log1p().alias("depth_log"),
        pl.col("_ret").std().fill_null(0.0).alias("rv"),
        pl.col("transaction_volume").sum().log1p().alias("txv_log"),
        pl.col("transaction_count").sum().log1p().alias("txc_log"),
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
    ret = np.where((prev > 0) & (~is_first), (ml - prev) / prev, 0.0)
    tensor[rows, 7, tcol] = ret[ok]
    print(f"slow tensor: {tensor.shape} ({time.time()-t0:.0f}s)", flush=True)
    return tensor

# ============ 2. TCN 模型 ============
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

# ============ 3. 主流程 ============
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    print("loading label...", flush=True)
    label = pl.read_ipc(f"{DATA}/train/label.feather", memory_map=False)
    tr_ids = label.filter(pl.col("month") <= 32)["sample_id"].to_numpy()
    va_ids = label.filter((pl.col("month") > 32) & (pl.col("month") <= 70))["sample_id"].to_numpy()
    print(f"train {len(tr_ids):,} valid {len(va_ids):,}", flush=True)

    print("building train tensors...", flush=True)
    Xf_tr = build_fast_tensor("train", tr_ids)
    Xs_tr = build_slow_tensor("train", tr_ids)
    y_tr = label.filter(pl.col("month") <= 32)["target"].to_numpy().astype(np.float32)
    print("building valid tensors...", flush=True)
    Xf_va = build_fast_tensor("train", va_ids)
    Xs_va = build_slow_tensor("train", va_ids)
    y_va = label.filter((pl.col("month") > 32) & (pl.col("month") <= 70))["target"].to_numpy().astype(np.float32)
    np.savez(f"{OUT}/p12_tensors.npz", Xf_tr=Xf_tr, Xs_tr=Xs_tr, Xf_va=Xf_va, Xs_va=Xs_va)

    model = DualTower().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    EPOCHS = 25
    BATCH = 512
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.MSELoss()
    y_mean, y_std = y_tr.mean(), y_tr.std() + 1e-6
    y_n = (y_tr - y_mean) / y_std

    Xft = torch.from_numpy(Xf_tr).to(device)
    Xst = torch.from_numpy(Xs_tr).to(device)
    Xfv = torch.from_numpy(Xf_va).to(device)
    Xsv = torch.from_numpy(Xs_va).to(device)
    n = len(y_tr)

    best_c, best_state = -1, None
    T0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            opt.zero_grad()
            loss = lossf(model(Xft[idx], Xst[idx]), torch.from_numpy(y_n[idx]).to(device))
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xfv, Xsv).cpu().numpy() * y_std + y_mean
        c = cos_uncenter(pv, y_va)
        if c > best_c:
            best_c = c
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"ep{ep+1}: loss={tot/n:.6f} valid_cos={c:.6f} ({time.time()-T0:.0f}s)", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_va = model(Xfv, Xsv).cpu().numpy() * y_std + y_mean
    np.savez(f"{OUT}/p12_valid_pred.npz", pred=p_va, y=y_va)
    print(f"\n=== P1-2 result ===\nbest valid_cos = {best_c:.6f} (PSEUDO fold, sequence-only)")
    print(f"tabular blend reference (PSEUDO): R0=0.1299, R2=0.1313, R2+micro=0.1349")
    print(f"saved {OUT}/p12_valid_pred.npz")

if __name__ == "__main__":
    main()
