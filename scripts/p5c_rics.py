# -*- coding: utf-8 -*-
"""P5-C — RICS Deterministic Cross-Channel Probe (任务书 2026-08-14, Step 5).

Question (§6): Market alpha 是否主要来自 short-range cross-channel geometry?
五层确定性消融 (R0-R4, 累计), 全部喂小 MLP + cosine loss:
  R0 flatten:        last-10 × 12ch = 120d
  R1 + moments:      mean/std/min/max/range/slope/last-first/recent-energy (96d)
  R2 + covariance:   zero-lag Cov + Corr upper-tri (132d)
  R3 + lag even/odd: r_ij(k)±r_ij(-k) /2, k=1,2,3 (396d)
  R4 + phase:        Re(ZiZj*), Im, cosΔφ, |sinΔφ| at f=1,2 + channel energy (552d)

对照: M0-ref = P5-01 200 步 Conv1D 原版 (corr(y)≈0.086 的当前 market 模型),
同一协议重训, 提供"current market prediction"参照 (预测未保存, 必须重训).

协议 (P5-01 复刻): train 21-40, alpha 41-50 选 (blend canonical baseline),
frozen 51-70。通道选择 (经济含义, 12): L1/L2 四价四量 + mid + spread1 +
depth1 + imb1。

时间反演 (§6.5-6.6): 对 frozen 窗口同模型 pred_fwd vs pred_rev (输入反演),
corr + MAE; p_even/p_odd 分解各自 cos(y) + blend。

判定 (§6.8-6.9):
  CONTINUE:  corr(p_RICS,y) ≥ 0.095 或 (corr(RICS,M0ref)<0.70 且 blendΔ≥+0.0006)
  STRONG:    corr > 0.10 且 late 稳定
  KILL:      corr < 0.09 或 corr(RICS,M0ref)>0.90 且 blend≈0 或 phase≤covariance
             或 late 恶化
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

MARKET = Path(r"D:\mscapital-forecasting\data\raw\train\market.feather")
LABEL = Path(r"D:\mscapital-forecasting\data\raw\train\label.feather")
CANON = Path(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
OUT = Path(r"D:\mscapital-kaggle\output\p5c_rics")
OUT.mkdir(parents=True, exist_ok=True)

STEPS = 10
NCH = 12
EPOCHS = 15
BATCH = 1024
LR = 1e-3
WD = 1e-4
SEED = 42
ALPHA_GRID = (0.0, 0.03, 0.05, 0.08, 0.10, 0.13, 0.17, 0.25)

# 12 core channels (经济含义选择, 非暴力搜索): L1/L2 price+size, mid, spread1, depth1, imb1
CH_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14]  # in 18-ch order
CH_NAMES = ["bid1", "ask1", "bv1", "av1", "bid2", "ask2", "bv2", "av2",
            "avgprice", "mid", "spread1", "depth1", "imb1"][:12]


def cosine(p: np.ndarray, y: np.ndarray) -> float:
    num = float(np.dot(p, y))
    den = float(np.sqrt(np.dot(p, p) * np.dot(y, y)))
    return num / den if den > 0 else 0.0


def rms_norm(p: np.ndarray) -> np.ndarray:
    s = float(np.sqrt(np.mean(p ** 2)))
    return p / s if s > 0 else p


# ---------------- last-10 grid tensor (P5-01 builder, 只取最后 10 个网格点) ----------------

def build_last10(market_path: Path, sample_ids: np.ndarray) -> np.ndarray:
    n = len(sample_ids)
    tmp = OUT / "seq10_tmp.bin"
    X = np.memmap(tmp, dtype=np.float32, mode="w+", shape=(n, STEPS, 18))
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
    grid = np.linspace(0, 600 - 600 / 200, 200)[-STEPS:]  # last 10 of the 200-grid
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
    return np.array(X)[:, :, CH_IDX]


# ---------------- feature layers (vectorized) ----------------

def feats_R0(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1)


def feats_R1(X: np.ndarray) -> np.ndarray:
    n = len(X)
    mu = X.mean(axis=1)
    sd = X.std(axis=1) + 1e-9
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    rng = mx - mn
    slope = (X[:, -1, :] - X[:, 0, :]) / (STEPS - 1)
    lf = X[:, -1, :] - X[:, 0, :]
    e_last = (X[:, -3:, :] ** 2).sum(axis=1)
    e_all = (X ** 2).sum(axis=1) + 1e-12
    en = e_last / e_all
    return np.hstack([X.reshape(n, -1), mu, sd, mn, mx, rng, slope, lf, en])


def feats_R2(X: np.ndarray) -> np.ndarray:
    """+ zero-lag Cov & Corr upper triangles."""
    n = len(X)
    Xz = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-9)
    cov = np.einsum("nti,ntj->nij", X - X.mean(axis=1, keepdims=True), X - X.mean(axis=1, keepdims=True)) / STEPS
    corr = np.einsum("nti,ntj->nij", Xz, Xz) / STEPS
    iu = np.triu_indices(NCH, 1)
    return np.hstack([feats_R1(X), cov[:, iu[0], iu[1]], corr[:, iu[0], iu[1]]])


def feats_R3(X: np.ndarray) -> np.ndarray:
    """+ symmetric/odd lagged cross-correlation, k=1,2,3."""
    n = len(X)
    Xz = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-9)
    iu = np.triu_indices(NCH, 1)
    parts = []
    for k in (1, 2, 3):
        A = Xz[:, :-k, :] if k else Xz
        B = Xz[:, k:, :] if k else Xz
        T = STEPS - k
        M = np.einsum("nti,ntj->nij", A, B) / T          # r_ij(k)
        Me = (M + M.transpose(0, 2, 1)) / 2              # even
        Mo = (M - M.transpose(0, 2, 1)) / 2              # odd
        parts.append(Me[:, iu[0], iu[1]])
        parts.append(Mo[:, iu[0], iu[1]])
    return np.hstack([feats_R2(X), *parts])


def feats_R4(X: np.ndarray) -> np.ndarray:
    """+ phase: Re/Im/cos/|sin| of Zi Zj* at f=1,2 + channel energy."""
    n = len(X)
    Xf = np.fft.rfft(X, axis=1).transpose(0, 2, 1)       # (n,12,6) freq per channel
    Z = Xf[:, :, 1:3]                                    # f=1,2
    Zc = Z.conj()
    cross = Z[:, :, None, :] * Zc[:, None, :, :]         # (n,12,12,2)
    iu = np.triu_indices(NCH, 1)
    csel = cross[:, iu[0], iu[1], :]                     # (n,66,2)
    amp = np.abs(Z)                                            # (n,12,2)
    ampij = amp[:, iu[0], :] * amp[:, iu[1], :]                # (n,66,2) |Zi||Zj| per freq
    denom = ampij + 1e-12
    Re = csel.real
    Im = csel.imag
    cosph = Re / denom
    sinph = np.abs(Im) / denom
    # channel energy + energy ratio
    pw = np.abs(Z) ** 2                                   # (n,12,2)
    pw_sum = pw.sum(axis=1, keepdims=True) + 1e-12
    ratio = pw / pw_sum
    return np.hstack([feats_R3(X), Re.reshape(n, -1), Im.reshape(n, -1),
                      cosph.reshape(n, -1), sinph.reshape(n, -1),
                      pw.reshape(n, -1), ratio.reshape(n, -1)])


LAYERS = {"R0": feats_R0, "R1": feats_R1, "R2": feats_R2, "R3": feats_R3, "R4": feats_R4}


# ---------------- models ----------------

class SmallMLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.SiLU(),
                                 nn.Linear(256, 256), nn.SiLU(),
                                 nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).reshape(-1)


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
    def __init__(self, ch: int = 18, steps: int = 200):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(ch, 64, 7, padding=3, bias=False), nn.BatchNorm1d(64), nn.GELU(),
        )
        self.blocks = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.head = nn.Sequential(nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x):
        h = self.embed(x.permute(0, 2, 1))
        h = self.blocks(h)
        return self.head(h.mean(dim=2))


def train_mlp(Xtr: np.ndarray, ytr: np.ndarray, device: torch.device) -> nn.Module:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = SmallMLP(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
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
            loss = 1.0 - torch.nn.functional.cosine_similarity(
                p.reshape(1, -1), yt[idx].reshape(1, -1), dim=1).squeeze()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        if ep % 5 == 0 or ep == EPOCHS:
            print(f"    [ep {ep:02d}/{EPOCHS}] loss={ep_loss / nb:.5f}", flush=True)
    return model


def train_seq(Xtr: np.ndarray, ytr: np.ndarray, device: torch.device) -> nn.Module:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = MarketSeqNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
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
            loss = 1.0 - torch.nn.functional.cosine_similarity(
                p.reshape(1, -1), yt[idx].reshape(1, -1), dim=1).squeeze()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        if ep % 5 == 0 or ep == EPOCHS:
            print(f"    [seq ep {ep:02d}/{EPOCHS}] loss={ep_loss / nb:.5f}", flush=True)
    return model


@torch.no_grad()
def predict_mlp(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), BATCH * 4):
        xb = torch.from_numpy(X[i:i + BATCH * 4]).float().to(device)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs).reshape(-1)


@torch.no_grad()
def predict_seq(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), BATCH * 4):
        xb = torch.from_numpy(X[i:i + BATCH * 4]).float().to(device)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs).reshape(-1)


# ---------------- eval ----------------

def eval_model(p_sel, y_sel, p_fr, y_fr, p_canon_sel, p_canon_fr, name):
    p_n_sel, _ = rms_norm(p_sel), None
    c_n_sel, _ = rms_norm(p_canon_sel), None
    best_a, best_s = 0.0, -1e9
    for a in ALPHA_GRID:
        s = cosine(c_n_sel + a * p_n_sel, y_sel)
        if s > best_s:
            best_s, best_a = s, a
    p_n_fr = rms_norm(p_fr)
    c_n_fr = rms_norm(p_canon_fr)
    s0 = cosine(c_n_fr, y_fr)
    s1 = cosine(c_n_fr + best_a * p_n_fr, y_fr)
    cy = float(np.corrcoef(p_fr, y_fr)[0, 1])
    cc = float(np.corrcoef(p_fr, p_canon_fr)[0, 1])
    md = []
    m_fr = None
    return {"alpha": float(best_a), "corr_y": cy, "corr_canon": cc,
            "delta_frozen": float(s1 - s0), "monthly": md, "m_fr": m_fr}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    t0 = time.time()

    lab = pl.read_ipc(LABEL)
    m_all = lab["month"].to_numpy()
    sid_all = lab["sample_id"].to_numpy()
    y_all = lab["target"].to_numpy()

    tr_mask = (m_all >= 21) & (m_all <= 40)
    sel_mask = (m_all >= 41) & (m_all <= 50)
    fr_mask = (m_all >= 51) & (m_all <= 70)
    va_mask = sel_mask | fr_mask

    canon = np.load(CANON)
    cids = canon["sample_id"]
    p_canon_sel = canon["baseline_oof"][np.searchsorted(cids, sid_all[sel_mask])]
    p_canon_fr = canon["baseline_oof"][np.searchsorted(cids, sid_all[fr_mask])]
    y_sel, y_fr = y_all[sel_mask], y_all[fr_mask]
    m_fr = m_all[fr_mask]

    # ---- last-10 tensor ----
    print("building last-10 sequences (21-40)...", flush=True)
    X10_tr = build_last10(MARKET, sid_all[tr_mask])
    print("building last-10 sequences (41-70)...", flush=True)
    X10_va = build_last10(MARKET, sid_all[va_mask])

    results = {"layers": {}, "reversal": {}, "decision": {}}
    p_ref = {}
    resume_path = OUT / "p5c_layer_preds.npz"
    saved = dict(np.load(resume_path)) if resume_path.exists() else {}
    saved_r4 = None

    for lname, ffn in LAYERS.items():
        print(f"\n===== layer {lname} =====", flush=True)
        if f"p_sel_{lname}" in saved:
            p_sel = saved[f"p_sel_{lname}"]
            p_fr = saved[f"p_fr_{lname}"]
            print(f"  {lname} resumed from disk (dims {len(p_fr):,})", flush=True)
        else:
            Ftr = ffn(X10_tr).astype(np.float32)
            Fva = ffn(X10_va).astype(np.float32)
            print(f"  {lname} dims: train {Ftr.shape}", flush=True)
            # standardize (train stats)
            mu = Ftr.mean(axis=0, keepdims=True)
            sd = Ftr.std(axis=0, keepdims=True) + 1e-6
            Ftr_s = (Ftr - mu) / sd
            Fva_s = (Fva - mu) / sd
            model = train_mlp(Ftr_s, y_all[tr_mask], device)
            p_va = predict_mlp(model, Fva_s, device)
            p_sel = p_va[sel_mask[va_mask]]
            p_fr = p_va[fr_mask[va_mask]]
            saved[f"p_sel_{lname}"] = p_sel
            saved[f"p_fr_{lname}"] = p_fr
            saved[f"dims_{lname}"] = int(Ftr.shape[1])
            np.savez(resume_path, **saved)
            if lname == "R4":
                saved_r4 = (model, mu.copy(), sd.copy())  # reuse for reversal (no retrain)
            del Ftr, Fva, Ftr_s, Fva_s, model
            gc.collect()
        p_ref[lname] = p_fr

        # eval (blend with canonical baseline, alpha on 41-50)
        p_n_sel = rms_norm(p_sel)
        c_n_sel = rms_norm(p_canon_sel)
        best_a, best_s = 0.0, -1e9
        for a in ALPHA_GRID:
            s = cosine(c_n_sel + a * p_n_sel, y_sel)
            if s > best_s:
                best_s, best_a = s, a
        p_n_fr = rms_norm(p_fr)
        c_n_fr = rms_norm(p_canon_fr)
        s0 = cosine(c_n_fr, y_fr)
        s1 = cosine(c_n_fr + best_a * p_n_fr, y_fr)
        cy = float(np.corrcoef(p_fr, y_fr)[0, 1])
        cc = float(np.corrcoef(p_fr, p_canon_fr)[0, 1])
        md = []
        for mm in range(51, 71):
            mmask = m_fr == mm
            if mmask.sum() > 100:
                md.append(float(cosine(c_n_fr[mmask] + best_a * p_n_fr[mmask], y_fr[mmask]) -
                                cosine(c_n_fr[mmask], y_fr[mmask])))
        md = np.asarray(md)
        d_late = float(md[-10:].mean())
        results["layers"][lname] = {
            "dims": int(saved.get(f"dims_{lname}", 0)),
            "corr_y": cy, "corr_canon": cc, "alpha": float(best_a),
            "delta_frozen": float(s1 - s0), "delta_late_61_70": d_late,
            "monthly_pos": int((md > 0).sum()), "monthly_n": int(len(md)),
            "monthly_mean": float(md.mean()),
        }
        print(f"  {lname}: corr_y={cy:+.4f} corr_canon={cc:.3f} alpha={best_a} "
              f"Δ={s1 - s0:+.6f} late={d_late:+.6f} months={(md > 0).sum()}/{len(md)}", flush=True)

    # ---- M0-ref: 200-step conv (current market model, P5-01 arch) ----
    print("\n===== M0-ref (200-step, P5-01 arch) =====", flush=True)
    m0 = None
    if "p_sel_M0" in saved:
        p_m0_sel = saved["p_sel_M0"]
        p_m0_fr = saved["p_fr_M0"]
        print(f"  M0 resumed from disk ({len(p_m0_fr):,})", flush=True)
    else:
        X200_tr = build_200(MARKET, sid_all[tr_mask], OUT / "seq200_tmp.bin")
        mu200 = X200_tr.reshape(-1, 18).mean(axis=0, keepdims=True).astype(np.float32)
        sd200 = X200_tr.reshape(-1, 18).std(axis=0, keepdims=True).astype(np.float32) + 1e-6
        X200_tr_s = (X200_tr - mu200) / sd200
        m0 = train_seq(X200_tr_s, y_all[tr_mask], device)
        del X200_tr, X200_tr_s
        gc.collect()
        print("building 200-step sequences (41-70)...", flush=True)
        X200_va = build_200(MARKET, sid_all[va_mask], OUT / "seq200_tmp.bin")
        X200_va_s = (X200_va - mu200) / sd200
        p_m0_va = predict_seq(m0, X200_va_s, device)
        del X200_va
        gc.collect()
        p_m0_sel = p_m0_va[sel_mask[va_mask]]
        p_m0_fr = p_m0_va[fr_mask[va_mask]]
        saved["p_sel_M0"] = p_m0_sel
        saved["p_fr_M0"] = p_m0_fr
        np.savez(resume_path, **saved)
    p_ref["M0"] = p_m0_fr

    p_n_sel = rms_norm(p_m0_sel)
    c_n_sel = rms_norm(p_canon_sel)
    best_a, best_s = 0.0, -1e9
    for a in ALPHA_GRID:
        s = cosine(c_n_sel + a * p_n_sel, y_sel)
        if s > best_s:
            best_s, best_a = s, a
    p_n_fr = rms_norm(p_m0_fr)
    c_n_fr = rms_norm(p_canon_fr)
    s0 = cosine(c_n_fr, y_fr)
    s1 = cosine(c_n_fr + best_a * p_n_fr, y_fr)
    results["layers"]["M0"] = {
        "dims": 200 * 18,
        "corr_y": float(np.corrcoef(p_m0_fr, y_fr)[0, 1]),
        "corr_canon": float(np.corrcoef(p_m0_fr, p_canon_fr)[0, 1]),
        "alpha": float(best_a),
        "delta_frozen": float(s1 - s0),
    }
    print(f"  M0: corr_y={results['layers']['M0']['corr_y']:+.4f} "
          f"alpha={best_a} Δ={results['layers']['M0']['delta_frozen']:+.6f}", flush=True)

    # ---- time reversal decomposition (§6.5-6.6): R4 + M0, 同一 frozen model ----
    print("\n===== reversal decomposition =====", flush=True)
    if saved_r4 is None:
        # R4 从磁盘续跑时无模型 → 同种子确定性重训 (结果与已存预测一致)
        print("  R4 model not in memory — deterministic retrain for reversal", flush=True)
        F10_tr = LAYERS["R4"](X10_tr).astype(np.float32)
        mu_r4 = F10_tr.mean(axis=0, keepdims=True)
        sd_r4 = F10_tr.std(axis=0, keepdims=True) + 1e-6
        model_r4 = train_mlp(((F10_tr - mu_r4) / sd_r4), y_all[tr_mask], device)
        del F10_tr
        gc.collect()
    else:
        model_r4, mu_r4, sd_r4 = saved_r4
    Xfr_rev = np.ascontiguousarray(X10_va[fr_mask[va_mask]][:, ::-1, :])
    Ffr_rev = LAYERS["R4"](Xfr_rev).astype(np.float32)
    p_r4_rev = predict_mlp(model_r4, ((Ffr_rev - mu_r4) / sd_r4), device)
    del Xfr_rev, Ffr_rev, model_r4
    gc.collect()
    p_r4_fr = p_ref["R4"]
    rev_r4 = {
        "corr_fwd_rev": float(np.corrcoef(p_r4_fr, p_r4_rev)[0, 1]),
        "mae_fwd_rev": float(np.mean(np.abs(p_r4_fr - p_r4_rev))),
    }
    p_even = (p_r4_fr + p_r4_rev) / 2
    p_odd = (p_r4_fr - p_r4_rev) / 2
    c_n_fr = rms_norm(p_canon_fr)
    ce = cosine(rms_norm(p_even), y_fr)
    co = cosine(rms_norm(p_odd), y_fr)
    # even/odd blend: frozen 段小网格, 诊断用途 (明确标注: 非 nested, 不参与判定)
    best_we, best_se = 0.0, -1e9
    for w in np.arange(0.0, 0.35, 0.05):
        s = cosine(c_n_fr + w * rms_norm(p_even), y_fr)
        if s > best_se:
            best_se, best_we = s, float(w)
    best_wo, best_so = 0.0, -1e9
    for w in np.arange(0.0, 0.35, 0.05):
        s = cosine(c_n_fr + w * rms_norm(p_odd), y_fr)
        if s > best_so:
            best_so, best_wo = s, float(w)
    rev_r4["cos_even_y"] = ce
    rev_r4["cos_odd_y"] = co
    rev_r4["blend_even_delta"] = best_se - s0
    rev_r4["blend_odd_delta"] = best_so - s0
    rev_r4["best_w_even"] = best_we
    rev_r4["best_w_odd"] = best_wo
    results["reversal"]["R4"] = rev_r4
    print(f"  R4 reversal: corr(fwd,rev)={rev_r4['corr_fwd_rev']:.4f} "
          f"MAE={rev_r4['mae_fwd_rev']:.5f} cos_even={ce:+.4f} cos_odd={co:+.4f} "
          f"blend_evenΔ={best_se - s0:+.6f} blend_oddΔ={best_so - s0:+.6f}", flush=True)

    # M0 reversal (同一 21-40 模型, 反转 200 步输入; 续跑无模型则跳过)
    if m0 is not None and "p_fr_M0" in saved:
        X200_fr_rev = np.ascontiguousarray(X200_va_s[fr_mask[va_mask]][:, ::-1, :])
        p_m0_rev = predict_seq(m0, X200_fr_rev, device)
        del X200_fr_rev
        gc.collect()
        rev_m0 = {
            "corr_fwd_rev": float(np.corrcoef(p_m0_fr, p_m0_rev)[0, 1]),
            "mae_fwd_rev": float(np.mean(np.abs(p_m0_fr - p_m0_rev))),
            "cos_even_y": float(cosine(rms_norm((p_m0_fr + p_m0_rev) / 2), y_fr)),
            "cos_odd_y": float(cosine(rms_norm((p_m0_fr - p_m0_rev) / 2), y_fr)),
        }
        results["reversal"]["M0"] = rev_m0
        print(f"  M0 reversal: corr(fwd,rev)={rev_m0['corr_fwd_rev']:.4f} "
              f"MAE={rev_m0['mae_fwd_rev']:.5f} cos_even={rev_m0['cos_even_y']:+.4f} "
              f"cos_odd={rev_m0['cos_odd_y']:+.4f}", flush=True)
        del X200_va_s, m0
        gc.collect()
    else:
        print("  M0 reversal skipped (model not in memory)", flush=True)

    # ---- consolidated + decision (§6.8-6.9) ----
    L = results["layers"]
    best_layer = max(["R0", "R1", "R2", "R3", "R4"], key=lambda k: L[k]["corr_y"])
    corr_r4_m0 = float(np.corrcoef(p_ref["R4"], p_ref["M0"])[0, 1])
    blend_r4 = L["R4"]["delta_frozen"]
    d_late_r4 = L["R4"]["delta_late_61_70"]

    crit = {
        "corr(R4,y) ≥ 0.095": L["R4"]["corr_y"] >= 0.095,
        "或 corr(R4,M0) < 0.70 且 blendΔ ≥ +0.0006": (corr_r4_m0 < 0.70) and (blend_r4 >= 0.0006),
        "phase > covariance (R4 corr_y > R2 corr_y)": L["R4"]["corr_y"] > L["R2"]["corr_y"],
        "late 61-70 ≥ 0": d_late_r4 >= 0,
        "blend Δ > 0": blend_r4 > 0,
    }
    if L["R4"]["corr_y"] >= 0.095 or ((corr_r4_m0 < 0.70) and blend_r4 >= 0.0006):
        if L["R4"]["corr_y"] > 0.10 and d_late_r4 >= 0:
            decision = "RICS STRONG LIVE"
        else:
            decision = "RICS LIVE"
    elif L["R4"]["corr_y"] < 0.09 or (corr_r4_m0 > 0.90 and blend_r4 < 0.0004) or d_late_r4 < 0:
        decision = "RICS KILL"
    else:
        decision = "INCONCLUSIVE"
    results["consolidated"] = {
        "best_layer_by_corr": best_layer,
        "corr_R4_M0": corr_r4_m0,
        "blend_delta_R4": blend_r4,
        "layer_ladder": {k: {"corr_y": L[k]["corr_y"], "delta": L[k]["delta_frozen"]} for k in
                         ["R0", "R1", "R2", "R3", "R4", "M0"]},
    }
    results["decision"] = {"verdict": decision, "criteria": crit}

    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    np.savez(OUT / "p5c_preds.npz",
             p_r0=p_ref["R0"], p_r1=p_ref["R1"], p_r2=p_ref["R2"], p_r3=p_ref["R3"],
             p_r4=p_ref["R4"], p_m0=p_ref["M0"],
             y=y_fr, month=m_fr,
             canon=p_canon_fr,
             p_r4_rev=p_r4_rev, p_r4_even=p_even, p_r4_odd=p_odd)

    print("\n" + "=" * 78)
    for k in ["R0", "R1", "R2", "R3", "R4", "M0"]:
        print(f"  {k:<4} corr_y={L[k]['corr_y']:+.4f} Δ={L[k]['delta_frozen']:+.6f} "
              f"late={L[k].get('delta_late_61_70', float('nan')):+.6f}")
    print(f"  corr(R4,M0)={corr_r4_m0:.3f}  best_layer={best_layer}")
    print(f"  decision: {decision}")
    print("=" * 78)
    print(f"\ntotal {time.time()-t0:.0f}s → {OUT}")


def build_200(market_path: Path, sample_ids: np.ndarray, tmp_path: Path) -> np.ndarray:
    """P5-01 exact 200-step builder (18 channels, full grid)."""
    n = len(sample_ids)
    X = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=(n, 200, 18))
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
    grid = np.linspace(0, 600 - 600 / 200, 200)
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


if __name__ == "__main__":
    main()
