# -*- coding: utf-8 -*-
"""P5-02I: Information Audit - surrogate data tests on market 600s sequence.

Question: WHICH structure inside the market sequence carries the alpha that
P5-01 found (corr(y)=0.086, frozen Δ=+0.0009)? Destroy one structure at a time,
keep everything else identical, watch the signal disappear.

Strict shared protocol (user constraint #1): same architecture / seed / epochs /
optimizer / split / alpha-selection as P5-01 cosine arm. Only the input is
surgically altered. No tuning anywhere.

Arms (user constraint #2: block shuffle at 10/20/50):
  M0 RAW              baseline (P5-01 cosine arm replica)
  M1 SHUFFLE          per-sample random permutation of 200 steps (kills order)
  M2 REVERSE          time reversal (kills arrow-of-time, keeps path shape)
  M3 BLOCK10/20/50    per-sample block shuffle (kills long-range dependency)
  M4 DESYNC           per-sample per-channel circular shift (kills channel sync)
  M5 PHASE            FFT phase randomization (kills nonlinear morphology,
                      keeps amplitude spectrum)

Target probes (user constraint #3: diagnostic only, no tuning):
  P_sign     sign(y)          -> BCE -> AUC
  P_rank     rank-CDF(y)      -> reg -> corr(pred, rank)
  P_mag      |y|              -> reg -> corr(pred, |y|)
  P_extreme  |y| > p90(train) -> BCE -> AUC
  P_resid    sign(y - beta*v7)-> BCE -> AUC  (beta = OLS on 33-40)

Split (frozen, same as P5-01): months 21-40 train / 41-50 alpha select ONLY /
51-70 FROZEN. Probes P_sign/P_rank/P_mag/P_extreme train 21-40; P_resid trains
33-40 (v7_like coverage starts at 33).

Output: results.json + info-localization decision table.
"""
from __future__ import annotations

import gc
import json
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
OUT = Path(r"D:\mscapital-kaggle\output\p5_02i_info_audit")
OUT.mkdir(parents=True, exist_ok=True)

STEPS = 200
CHANNELS = 18
EPOCHS = 15
BATCH = 1024
LR = 1e-3
WD = 1e-4
SEED = 42
ALPHA_GRID = (0.0, 0.03, 0.05, 0.08, 0.10, 0.13, 0.17, 0.25)


def build_sequence(market_path: Path, sample_ids: np.ndarray, steps: int = STEPS) -> np.ndarray:
    """P5-01 exact replica: polars vectorized derived columns + memmap + per-sample
    time-resample to uniform grid. Channel order: [bid1, ask1, bv1, av1, bid2, ask2,
    bv2, av2, avgprice, vol, count, mid, spread1, depth1, imb1, spread2, depth2, imb2]."""
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
    """P5-01 exact architecture replica."""
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
        return self.head(h.mean(dim=2))


def train_arm(Xtr: np.ndarray, ytr: np.ndarray, device: torch.device,
              seed: int = SEED, head: str = "reg", y_is_cls: bool = False) -> nn.Module:
    """Shared training protocol: 15 ep, AdamW 1e-3, batch cosine (uncentered) for
    regression; BCE for classification probes. Same arch, head 1d either way."""
    torch.manual_seed(seed)
    np.random.seed(seed)
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
            if head == "cls":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(p.reshape(-1), yt[idx])
            else:
                loss = 1.0 - torch.nn.functional.cosine_similarity(
                    p.reshape(1, -1), yt[idx].reshape(1, -1), dim=1).squeeze()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        if ep % 5 == 0 or ep == EPOCHS:
            print(f"    [{head} ep {ep:02d}/{EPOCHS}] loss={ep_loss / nb:.5f}", flush=True)
    return model


@torch.no_grad()
def predict_surrogate(model: nn.Module, raw_va: np.ndarray, mu: np.ndarray, sd: np.ndarray,
                      arm: str, seed: int, device: torch.device) -> np.ndarray:
    """Predict with on-the-fly surrogate transform, blockwise to cap memory
    (raw_va stays resident 10GB; only one ~2.5GB transformed block at a time)."""
    model.eval()
    outs = []
    BLK = BATCH * 4
    for i in range(0, len(raw_va), BLK):
        Xb = raw_va[i:i + BLK] if arm == "raw" else transform(raw_va[i:i + BLK], arm, seed + i // BLK)
        Xb_s = (Xb - mu) / sd
        xb = torch.from_numpy(Xb_s).float().to(device)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs).reshape(-1)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), BATCH * 4):
        xb = torch.from_numpy(X[i:i + BATCH * 4]).float().to(device)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs).reshape(-1)


# ---------------- surrogate transforms (raw input -> raw output, same stats) ----------------

def transform(raw: np.ndarray, kind: str, seed: int) -> np.ndarray:
    n, steps, ch = raw.shape
    rng = np.random.default_rng(seed)
    if kind == "shuffle":
        perm = rng.permuted(np.broadcast_to(np.arange(steps), (n, steps)), axis=1)
        return np.take_along_axis(raw, np.broadcast_to(perm[:, :, None], (n, steps, ch)), axis=1)
    if kind == "reverse":
        return raw[:, ::-1, :]
    if kind.startswith("block"):
        bs = int(kind[5:])
        nb = steps // bs
        perm = rng.permuted(np.broadcast_to(np.arange(nb), (n, nb)), axis=1)
        idx = (perm[:, :, None] * bs + np.arange(bs)[None, None, :]).reshape(n, -1)
        return np.take_along_axis(raw, np.broadcast_to(idx[:, :, None], (n, steps, ch)), axis=1)
    if kind == "desync":
        shifts = rng.integers(0, steps, size=(n, ch))
        idx = (np.arange(steps)[None, None, :] - shifts[:, :, None]) % steps
        return np.take_along_axis(raw.transpose(0, 2, 1), idx, axis=2).transpose(0, 2, 1)
    if kind == "phase":
        # Blockwise FFT phase randomization. Keep float32/complex64 end-to-end:
        # python scalar `1j` promotes to complex128 (9.6GB alloc on full train).
        out = np.empty_like(raw)
        nrows = raw.shape[0] * ch
        flat = raw.reshape(-1, steps)
        BLK = 400_000
        for i in range(0, nrows, BLK):
            fb = flat[i:i + BLK]
            Xf = np.fft.rfft(fb, axis=1)                    # complex64
            amp = np.abs(Xf).astype(np.float32)             # float32
            ph = rng.uniform(0.0, 2.0 * np.pi, size=amp.shape).astype(np.float32)
            Xp = amp * np.exp(np.complex64(1j) * ph)        # complex64
            Xp[:, 0] = Xf[:, 0]                             # keep DC (mean)
            Xp[:, -1] = Xp[:, -1].real                      # Nyquist must stay real
            out.reshape(-1, steps)[i:i + BLK] = np.fft.irfft(Xp, n=steps, axis=1).astype(np.float32)
        return out
    raise ValueError(kind)


# ---------------- evaluation (P5-01 replica) ----------------

def evaluate_arm(p_sel: np.ndarray, y_sel: np.ndarray, v7_sel: np.ndarray,
                 p_fr: np.ndarray, y_fr: np.ndarray, v7_fr: np.ndarray,
                 act_fr: np.ndarray, m_fr: np.ndarray, name: str) -> dict:
    p_n_sel, _ = normalize_prediction(p_sel, "rms")
    v7_n_sel, _ = normalize_prediction(v7_sel, "rms")
    best_a, best_s = 0.0, -1e9
    for a in ALPHA_GRID:
        s = cosine_uncentered(v7_n_sel + a * p_n_sel, y_sel)
        if s > best_s:
            best_s, best_a = s, a
    p_n_fr, _ = normalize_prediction(p_fr, "rms")
    v7_n_fr, _ = normalize_prediction(v7_fr, "rms")
    s0 = cosine_uncentered(v7_n_fr, y_fr)
    s1 = cosine_uncentered(v7_n_fr + best_a * p_n_fr, y_fr)
    cy = float(np.corrcoef(p_fr, y_fr)[0, 1])
    cv = float(np.corrcoef(p_fr, v7_fr)[0, 1])
    res = {"alpha": float(best_a), "corr_y": cy, "corr_v7": cv,
           "delta_frozen": float(s1 - s0)}
    fin = v7_n_fr + best_a * p_n_fr
    for lo_, hi_, lbl in ((0.0, 0.5, "lo"), (0.5, 0.9, "mid"), (0.9, 1.0, "hi")):
        m2 = (act_fr >= np.quantile(act_fr, lo_)) & (act_fr <= np.quantile(act_fr, hi_))
        if m2.sum() < 500:
            res[f"delta_{lbl}"] = 0.0
            continue
        res[f"delta_{lbl}"] = float(cosine_uncentered(fin[m2], y_fr[m2]) -
                                    cosine_uncentered(v7_n_fr[m2], y_fr[m2]))
    md = []
    for m in range(51, 71):
        mm = (m_fr == m)
        if mm.sum() > 100:
            md.append(float(cosine_uncentered(fin[mm], y_fr[mm]) -
                            cosine_uncentered(v7_n_fr[mm], y_fr[mm])))
    md = np.asarray(md)
    res["monthly_pos"] = int((md > 0).sum())
    res["monthly_n"] = len(md)
    res["monthly_mean"] = float(md.mean())
    print(f"  [{name}] corr_y={cy:+.4f} corr_v7={cv:+.4f} alpha={best_a} "
          f"frozen_delta={s1 - s0:+.6f} monthly={res['monthly_pos']}/{res['monthly_n']} "
          f"lo={res['delta_lo']:+.6f} hi={res['delta_hi']:+.6f}", flush=True)
    return res


def auc(y_bin: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC without sklearn."""
    import scipy.stats
    r = scipy.stats.rankdata(score)
    n1 = int((y_bin == 1).sum())
    n0 = len(y_bin) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y_bin == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


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
    print(f"train(21-40)={tr_mask.sum():,} select(41-50)={sel_mask.sum():,} "
          f"frozen(51-70)={fr_mask.sum():,}", flush=True)

    # v7-like OOF baseline (same as P5-01: 0.8*v5 + 0.2*RL, rlps pseudo OOF 33-70)
    canon = np.load(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
    cids, cmonth = canon["sample_id"], canon["month"].astype(int)
    cbase = canon["baseline_oof"]
    rl = np.load(r"D:\mscapital-kaggle\output\rlps_final\realmlp_pseudo_pred.npz")["pred"]
    v5 = np.load(r"D:\mscapital-kaggle\output\rlps_v12\v5_table_pseudo_pred.npz")["pred"]
    fe_all = pl.read_parquet(r"D:\mscapital-forecasting\data\processed\f0726_train_f32.parquet").sort("sample_id")
    rl_ids = fe_all["sample_id"].to_numpy()[(m_all >= 33) & (m_all <= 70)]
    rp = np.searchsorted(rl_ids, cids)
    v7_like = 0.8 * v5[rp] + 0.2 * rl[rp]
    cpos = np.searchsorted(cids, sid_all[va_mask])
    v7_va = v7_like[cpos]
    y_sel, v7_sel = y_all[sel_mask], v7_va[sel_mask[va_mask]]
    y_fr, v7_fr = y_all[fr_mask], v7_va[fr_mask[va_mask]]

    # activity stratification (snapshot count per sample, frozen window)
    act = pl.read_ipc(MARKET, columns=["sample_id"]).group_by("sample_id").len().sort("sample_id")
    a_ids = act["sample_id"].to_numpy()
    a_cnt = act["len"].to_numpy().astype(np.float64)
    act_fr = a_cnt[np.searchsorted(a_ids, sid_all[fr_mask])]

    print("building raw sequences (21-40)...", flush=True)
    X_tr_raw = build_sequence(MARKET, sid_all[tr_mask])
    print("building raw sequences (41-70)...", flush=True)
    X_va_raw = build_sequence(MARKET, sid_all[va_mask])
    print(f"built {X_tr_raw.shape} / {X_va_raw.shape} in {time.time() - t0:.0f}s", flush=True)

    # shared standardization (train stats only, same for every arm)
    mu = X_tr_raw.reshape(-1, CHANNELS).mean(axis=0, keepdims=True).astype(np.float32)
    sd = X_tr_raw.reshape(-1, CHANNELS).std(axis=0, keepdims=True).astype(np.float32) + 1e-6

    results = {"arms": {}, "probes": {}}
    rj = OUT / "results.json"
    if rj.exists():
        old = json.loads(rj.read_text(encoding="utf-8"))
        results["arms"] = old.get("arms", {})
        results["probes"] = old.get("probes", {})
        print(f"resume: {len(results['arms'])} arms / {len(results['probes'])} probe keys already done", flush=True)
    arms = ["raw", "shuffle", "reverse", "block10", "block20", "block50", "desync", "phase"]
    for k, arm in enumerate(arms):
        if arm in results["arms"]:
            print(f"\n===== M{k}: {arm.upper()} (SKIP, already done) =====", flush=True)
            continue
        print(f"\n===== M{k}: {arm.upper()} =====", flush=True)
        if arm == "raw":
            Xa_s = (X_tr_raw - mu) / sd
        else:
            seed = SEED + 10 * (k + 1)
            Xa_s = (transform(X_tr_raw, arm, seed=seed) - mu) / sd
        model = train_arm(Xa_s, y_all[tr_mask], device)
        p_va = predict_surrogate(model, X_va_raw, mu, sd, arm, SEED + 10 * (k + 1) + 5, device)
        results["arms"][arm] = evaluate_arm(p_va[sel_mask[va_mask]], y_sel, v7_sel,
                                            p_va[fr_mask[va_mask]], y_fr, v7_fr, act_fr,
                                            m_all[fr_mask], arm)
        (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        del Xa_s, model
        gc.collect()

    # ---------------- target probes (same encoder, different head/target) ----------------
    print("\n===== TARGET PROBES =====", flush=True)
    X_tr_s = (X_tr_raw - mu) / sd
    X_va_s = (X_va_raw - mu) / sd
    probes = results["probes"]
    y_va = y_all[va_mask]  # align with X_va_s / masks (sel_mask[va_mask], fr_mask[va_mask])

    # P_sign
    if "sign_41-50_auc" not in probes:
        print("\n-- P_sign (sign(y)) --", flush=True)
        ys_tr = (y_all[tr_mask] > 0).astype(np.float32)
        model = train_arm(X_tr_s, ys_tr, device, head="cls")
        logit = predict(model, X_va_s, device)
        for lbl, mask in (("41-50", sel_mask[va_mask]), ("51-70", fr_mask[va_mask])):
            yb = (y_va[mask] > 0).astype(np.float32)
            probes[f"sign_{lbl}_auc"] = auc(yb, logit[mask])
            probes[f"sign_{lbl}_corr_y"] = float(np.corrcoef(logit[mask], y_va[mask])[0, 1])
        (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"    sign: {probes}", flush=True)
    else:
        print("-- P_sign SKIP (already done) --", flush=True)

    # P_rank (train-CDF mapped)
    if "rank_41-50_corr" not in probes:
        print("\n-- P_rank (rank-CDF) --", flush=True)
        sy = np.sort(y_all[tr_mask])
        rank_tr = np.searchsorted(sy, y_all[tr_mask]) / len(sy)
        model = train_arm(X_tr_s, rank_tr.astype(np.float32), device)
        pr = predict(model, X_va_s, device)
        for lbl, mask in (("41-50", sel_mask[va_mask]), ("51-70", fr_mask[va_mask])):
            rk = np.searchsorted(sy, y_va[mask]) / len(sy)
            probes[f"rank_{lbl}_corr"] = float(np.corrcoef(pr[mask], rk)[0, 1])
            probes[f"rank_{lbl}_corr_y"] = float(np.corrcoef(pr[mask], y_va[mask])[0, 1])
        (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"    rank: {probes}", flush=True)
    else:
        print("-- P_rank SKIP (already done) --", flush=True)

    # P_mag
    if "mag_41-50_corr" not in probes:
        print("\n-- P_mag (|y|) --", flush=True)
        model = train_arm(X_tr_s, np.abs(y_all[tr_mask]), device)
        pm = predict(model, X_va_s, device)
        for lbl, mask in (("41-50", sel_mask[va_mask]), ("51-70", fr_mask[va_mask])):
            probes[f"mag_{lbl}_corr"] = float(np.corrcoef(pm[mask], np.abs(y_va[mask]))[0, 1])
            probes[f"mag_{lbl}_corr_y"] = float(np.corrcoef(pm[mask], y_va[mask])[0, 1])
        (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"    mag: {probes}", flush=True)
    else:
        print("-- P_mag SKIP (already done) --", flush=True)

    # P_extreme
    if "extreme_41-50_auc" not in probes:
        print("\n-- P_extreme (|y| > p90 train) --", flush=True)
        thr = np.quantile(np.abs(y_all[tr_mask]), 0.90)
        ye_tr = (np.abs(y_all[tr_mask]) > thr).astype(np.float32)
        model = train_arm(X_tr_s, ye_tr, device, head="cls")
        pe = predict(model, X_va_s, device)
        for lbl, mask in (("41-50", sel_mask[va_mask]), ("51-70", fr_mask[va_mask])):
            yb = (np.abs(y_va[mask]) > thr).astype(np.float32)
            probes[f"extreme_{lbl}_auc"] = auc(yb, pe[mask])
            probes[f"extreme_{lbl}_corr_y"] = float(np.corrcoef(pe[mask], y_va[mask])[0, 1])
        (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"    extreme: {probes}", flush=True)
    else:
        print("-- P_extreme SKIP (already done) --", flush=True)

    # P_resid (sign of y - beta*v7; OLS beta on 33-40, v7_like coverage)
    if "resid_41-50_auc" not in probes:
        print("\n-- P_resid (sign(y - beta*v7)) --", flush=True)
        m3340 = (m_all >= 33) & (m_all <= 40)
        sid_3340 = sid_all[m3340]
        rpos = np.searchsorted(cids, sid_3340)
        v7_3340 = v7_like[rpos]
        y_3340 = y_all[m3340]
        beta = float(np.cov(y_3340, v7_3340)[0, 1] / np.var(v7_3340))
        r_tr = (y_3340 - beta * v7_3340 > 0).astype(np.float32)
        X_3340_s = (X_tr_raw[m_all[tr_mask] >= 33] - mu) / sd
        model = train_arm(X_3340_s, r_tr, device, head="cls")
        pr_ = predict(model, X_va_s, device)
        for lbl, mask in (("41-50", sel_mask[va_mask]), ("51-70", fr_mask[va_mask])):
            rr = (y_va[mask] - beta * v7_va[mask] > 0).astype(np.float32)
            probes[f"resid_{lbl}_auc"] = auc(rr, pr_[mask])
            probes[f"resid_{lbl}_corr_y"] = float(np.corrcoef(pr_[mask], y_va[mask])[0, 1])
        probes["beta"] = beta
        (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"    resid: {probes}", flush=True)
    else:
        print("-- P_resid SKIP (already done) --", flush=True)

    # ---------------- decision table ----------------
    print("\n=== P5-02I INFORMATION-LOCALIZATION DECISION TABLE ===")
    A = results["arms"]
    c0 = A["raw"]["corr_y"]
    d0 = A["raw"]["delta_frozen"]
    rows = [("M0 raw", "baseline", A["raw"])]
    for k, arm in enumerate(arms[1:], 1):
        rows.append((f"M{k} {arm}", "", A[arm]))
    print(f"{'arm':<16}{'corr_y':>8}{'R(corr_y)':>10}{'frozen_d':>10}{'R(frozen)':>10}"
          f"{'corr_v7':>9}{'monthly':>9}{'lo/hi':>13}")
    for name, _, r in rows:
        r_c = (c0 - r["corr_y"]) / c0 if c0 != 0 else float("nan")
        r_d = (d0 - r["delta_frozen"]) / d0 if d0 != 0 else float("nan")
        print(f"{name:<16}{r['corr_y']:>+8.4f}{r_c:>10.2f}{r['delta_frozen']:>+10.6f}"
              f"{r_d:>10.2f}{r['corr_v7']:>+9.3f}{r['monthly_pos']:>6}/{r['monthly_n']:<2}"
              f"{r['delta_lo']:>+7.4f}/{r['delta_hi']:>+6.4f}")
    print(f"\nprobes: {json.dumps(probes, indent=2)}")
    print(f"\ntotal time {time.time() - t0:.0f}s. results written to {OUT}")


if __name__ == "__main__":
    main()
