# -*- coding: utf-8 -*-
"""P5-01: Market-only sequence baseline (600s trajectory -> target).

Information-existence test for market.feather as a NEW data source.
NOTE: LB142 channel mapping is UNKNOWN - this is OUR minimum implementation:
  11 raw fields (bid/ask p+v L1/L2, tx avgprice/vol/count)
  + 7 derived (mid, spread1, depth1, imb1, spread2, depth2, imb2)

Design (strict, no P4-08A mistake):
- sequence: 200 steps x 18 channels, time-resampled to uniform 3s grid
  (0..597s, carry-forward last snapshot)
- architecture: small residual Conv1D (64ch, k=7, 2 residual blocks) -> GAP -> 32d -> 1
- arms: A market-only MSE, B market-only cosine (uncentered), same seed/arch/prep
- split: months 21-40 TRAIN (structure/epochs), 41-50 alpha select ONLY,
         51-70 FROZEN final validation (nothing tuned there)
- outputs: corr(market, v7), corr(market, residual_v7), v7+alpha*market delta
  on 51-70, activity-stratified deltas
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from mscapital.metrics import cosine_uncentered, normalize_prediction

MARKET = Path(r"D:\mscapital-forecasting\data\raw\train\market.feather")
LABEL = Path(r"D:\mscapital-forecasting\data\raw\train\label.feather")
V7 = Path(r"D:\mscapital-kaggle\output\submissions\submission_blend_v7_rl20.csv")  # test; not used for OOF
OUT = Path(r"D:\mscapital-kaggle\output\p5_01_market_sequence")
OUT.mkdir(parents=True, exist_ok=True)

STEPS = 200
CHANNELS = 18
EPOCHS = 15
BATCH = 1024
LR = 1e-3
SEED = 42


def build_sequence(market_path: Path, sample_ids: np.ndarray, steps: int = STEPS) -> np.ndarray:
    """Time-resample snapshots to uniform grid; carry-forward last value.
    Streaming: polars lazy filter by sample_id chunk + numpy per-sample loop."""
    n = len(sample_ids)
    tmp = OUT / "seq_tmp.bin"
    X = np.memmap(tmp, dtype=np.float32, mode="w+", shape=(n, steps, CHANNELS))
    cols_in = [
        "sample_id", "seconds_before_predict", "bid_price_1", "ask_price_1",
        "bid_volume_1", "ask_volume_1", "bid_price_2", "ask_price_2",
        "bid_volume_2", "ask_volume_2", "transaction_avgprice",
        "transaction_volume", "transaction_count",
    ]
    cols = ["bid_price_1", "ask_price_1", "bid_volume_1", "ask_volume_1",
            "bid_price_2", "ask_price_2", "bid_volume_2", "ask_volume_2",
            "transaction_avgprice", "transaction_volume", "transaction_count",
            "mid", "spread1", "depth1", "imb1", "spread2", "depth2", "imb2"]
    grid = np.linspace(0, 600 - 600 / steps, steps)
    lf = pl.scan_ipc(market_path)
    lf = lf.with_columns([
        ((pl.col("bid_price_1") + pl.col("ask_price_1")) / 2).alias("mid"),
        (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("spread1"),
        (pl.col("bid_volume_1") + pl.col("ask_volume_1")).alias("depth1"),
        ((pl.col("bid_volume_1") - pl.col("ask_volume_1")) / (pl.col("bid_volume_1") + pl.col("ask_volume_1") + 1e-6)).alias("imb1"),
        (pl.col("ask_price_2") - pl.col("bid_price_2")).alias("spread2"),
        (pl.col("bid_volume_2") + pl.col("ask_volume_2")).alias("depth2"),
        ((pl.col("bid_volume_2") - pl.col("ask_volume_2")) / (pl.col("bid_volume_2") + pl.col("ask_volume_2") + 1e-6)).alias("imb2"),
    ]).select(cols_in[:2] + cols)
    t0 = time.time()
    lo, hi = int(sample_ids.min()), int(sample_ids.max())
    chunk = 200_000
    done = 0
    for a in range(lo, hi + 1, chunk):
        b = min(a + chunk - 1, hi)
        df = lf.filter(pl.col("sample_id").is_between(a, b)).collect()
        if df.height == 0:
            continue
        df = df.sort(["sample_id", "seconds_before_predict"])
        sid = df["sample_id"].to_numpy()
        sec = df["seconds_before_predict"].to_numpy()
        vals = df.select(cols).to_numpy()
        # map chunk rows to output rows
        i0 = np.searchsorted(sample_ids, sid[0], side="left")
        i1 = np.searchsorted(sample_ids, sid[-1], side="right")
        starts = np.searchsorted(sid, sample_ids[i0:i1], side="left")
        for k, s in enumerate(sample_ids[i0:i1]):
            lo2 = starts[k]
            hi2 = len(sid) if k + 1 >= len(sample_ids[i0:i1]) else np.searchsorted(sid, sample_ids[i0:i1][k + 1], side="left")
            if lo2 >= hi2:
                continue
            sv, vv = sec[lo2:hi2], vals[lo2:hi2]
            pos = np.clip(np.searchsorted(sv, grid, side="right") - 1, 0, len(sv) - 1)
            X[i0 + k] = np.nan_to_num(vv[pos], nan=0.0).astype(np.float32)
        done += i1 - i0
        print(f"  built {done:,}/{n:,} ({time.time()-t0:.0f}s)", flush=True)
    X.flush()
    return np.array(X)


class ResidualBlock(nn.Module):
    def __init__(self, c: int, k: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c, c, k, padding=k // 2, bias=False), nn.BatchNorm1d(c), nn.GELU(),
            nn.Conv1d(c, c, k, padding=k // 2, bias=False), nn.BatchNorm1d(c),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class MarketSeqNet(nn.Module):
    def __init__(self, ch: int = CHANNELS, steps: int = STEPS):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(ch, 64, 7, padding=3, bias=False), nn.BatchNorm1d(64), nn.GELU(),
        )
        self.blocks = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.head = nn.Sequential(nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x):
        h = self.embed(x.permute(0, 2, 1))  # (B, steps, ch) -> (B, ch, steps)
        h = self.blocks(h)
        h = h.mean(dim=2)          # GAP -> (B, 64)
        return self.head(h).reshape(-1)


def train_arm(Xtr: np.ndarray, ytr: np.ndarray, loss_name: str, device: torch.device) -> MarketSeqNet:
    torch.manual_seed(SEED)
    model = MarketSeqNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    Xt = torch.from_numpy(Xtr).float().to(device)
    yt = torch.from_numpy(ytr.astype(np.float32)).to(device)
    n = len(Xtr)
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        nb = 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            p = model(Xt[idx])
            if loss_name == "mse":
                loss = torch.nn.functional.mse_loss(p, yt[idx])
            else:
                loss = 1.0 - torch.nn.functional.cosine_similarity(
                    p.reshape(1, -1), yt[idx].reshape(1, -1), dim=1).squeeze()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        if ep % 5 == 0 or ep == EPOCHS:
            print(f"  [{loss_name} ep {ep:02d}/{EPOCHS}] loss={ep_loss/nb:.5f}", flush=True)
    return model


@torch.no_grad()
def predict(model: MarketSeqNet, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), BATCH * 4):
        xb = torch.from_numpy(X[i:i + BATCH * 4]).float().to(device)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    lab = pl.read_ipc(LABEL)
    m_all = lab["month"].to_numpy()
    sid_all = lab["sample_id"].to_numpy()
    y_all = lab["target"].to_numpy()

    tr_mask = (m_all >= 21) & (m_all <= 40)
    sel_mask = (m_all >= 41) & (m_all <= 50)
    fr_mask = (m_all >= 51) & (m_all <= 70)
    print(f"train(21-40)={tr_mask.sum():,} select(41-50)={sel_mask.sum():,} frozen(51-70)={fr_mask.sum():,}")

    # canonical OOF baseline for the same rows (CleanBaseline v2 OOF)
    canon = np.load(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
    cids, cmonth = canon["sample_id"], canon["month"].astype(int)
    cbase = canon["baseline_oof"]
    # v7-like OOF: use rlps pseudo OOF (33-70) blended 0.8/0.2 (approximation, same as P4-08E)
    rl = np.load(r"D:\mscapital-kaggle\output\rlps_final\realmlp_pseudo_pred.npz")["pred"]
    v5 = np.load(r"D:\mscapital-kaggle\output\rlps_v12\v5_table_pseudo_pred.npz")["pred"]
    fe_all = pl.read_parquet(r"D:\mscapital-forecasting\data\processed\f0726_train_f32.parquet").sort("sample_id")
    rl_ids = fe_all["sample_id"].to_numpy()[(m_all >= 33) & (m_all <= 70)]
    rp = np.searchsorted(rl_ids, cids)
    rl_c, v5_c = rl[rp], v5[rp]
    v7_like = 0.8 * v5_c + 0.2 * rl_c  # same formula as production (43 script)

    # build sequences: train 21-40 (needed for fit), then select+frozen together 41-70
    print("building train sequences (21-40)...", flush=True)
    X_tr = build_sequence(MARKET, sid_all[tr_mask])
    print("building valid sequences (41-70)...", flush=True)
    va_mask = sel_mask | fr_mask
    X_va = build_sequence(MARKET, sid_all[va_mask])

    # per-channel standardization (train stats only)
    mu = X_tr.reshape(-1, CHANNELS).mean(axis=0, keepdims=True).astype(np.float32)
    sd = X_tr.reshape(-1, CHANNELS).std(axis=0, keepdims=True).astype(np.float32) + 1e-6
    X_tr_s = (X_tr.astype(np.float32) - mu) / sd
    X_va_s = (X_va.astype(np.float32) - mu) / sd

    results = {}
    for loss_name in ("mse", "cosine"):
        print(f"\n===== market-only {loss_name} =====")
        model = train_arm(X_tr_s, y_all[tr_mask], loss_name, device)
        p_va = predict(model, X_va_s, device)
        # alpha selection on 41-50 only
        p_sel = p_va[sel_mask[va_mask]]
        p_fr = p_va[fr_mask[va_mask]]
        # v7_like aligned to valid rows (41-70 subset of canonical)
        cpos = np.searchsorted(cids, sid_all[va_mask])
        v7_va = v7_like[cpos]
        y_sel, v7_sel = y_all[sel_mask], v7_va[sel_mask[va_mask]]
        y_fr, v7_fr = y_all[fr_mask], v7_va[fr_mask[va_mask]]
        p_n_sel, _ = normalize_prediction(p_sel, "rms")
        v7_n_sel, _ = normalize_prediction(v7_sel, "rms")
        best_a, best_s = 0.0, -1e9
        for a in (0.0, 0.03, 0.05, 0.08, 0.10, 0.13, 0.17, 0.25):
            s = cosine_uncentered(v7_n_sel + a * p_n_sel, y_sel)
            if s > best_s:
                best_s, best_a = s, a
        print(f"  alpha selected on 41-50: {best_a} (score {best_s:.6f})")
        # FROZEN 51-70
        p_n_fr, _ = normalize_prediction(p_fr, "rms")
        v7_n_fr, _ = normalize_prediction(v7_fr, "rms")
        s0 = cosine_uncentered(v7_n_fr, y_fr)
        s1 = cosine_uncentered(v7_n_fr + best_a * p_n_fr, y_fr)
        s1b = cosine_uncentered(v7_n_fr + 0.13 * p_n_fr, y_fr)
        print(f"  FROZEN 51-70: v7 baseline={s0:.6f} | +a*market={s1:.6f} (delta {s1-s0:+.6f}) | +0.13: {s1b-s0:+.6f}")
        # corr diagnostics
        print(f"  corr(market, v7_like)      = {np.corrcoef(p_fr, v7_fr)[0,1]:+.4f}")
        r_v7 = y_fr - v7_fr
        print(f"  corr(market, residual_v7)  = {np.corrcoef(p_fr, r_v7)[0,1]:+.4f}")
        # market-only standalone signal
        print(f"  corr(market, y)            = {np.corrcoef(p_fr, y_fr)[0,1]:+.4f}")
        # activity stratification on 51-70 (market snapshot count)
        act = pl.read_ipc(MARKET, columns=["sample_id"]).group_by("sample_id").len().sort("sample_id")
        a_ids = act["sample_id"].to_numpy()
        a_cnt = act["len"].to_numpy().astype(np.float64)
        apos = np.searchsorted(a_ids, sid_all[fr_mask])
        act_fr = a_cnt[apos]
        fin = v7_n_fr + best_a * p_n_fr
        for lo_, hi_, lbl in ((0.0, 0.5, "lo"), (0.5, 0.9, "mid"), (0.9, 1.0, "hi")):
            m2 = (act_fr >= np.quantile(act_fr, lo_)) & (act_fr <= np.quantile(act_fr, hi_))
            d = cosine_uncentered(fin[m2], y_fr[m2]) - cosine_uncentered(v7_n_fr[m2], y_fr[m2])
            print(f"  {lbl}-activity frozen delta: {d:+.6f}")
        # monthly
        md = []
        for m in range(51, 71):
            mm = (m_all[fr_mask] == m)
            if mm.sum() > 100:
                d = cosine_uncentered(fin[mm], y_fr[mm]) - cosine_uncentered(v7_n_fr[mm], y_fr[mm])
                md.append(d)
        md = np.asarray(md)
        print(f"  monthly: pos={int((md > 0).sum())}/{len(md)} mean={md.mean():+.6f}")
        results[loss_name] = {"alpha": best_a, "delta_frozen": float(s1 - s0),
                              "delta_frozen_013": float(s1b - s0),
                              "corr_v7": float(np.corrcoef(p_fr, v7_fr)[0, 1]),
                              "corr_resid": float(np.corrcoef(p_fr, r_v7)[0, 1])}

    import json
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== P5-01 summary ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("written to", OUT)


if __name__ == "__main__":
    main()
