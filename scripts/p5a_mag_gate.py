# -*- coding: utf-8 -*-
"""P5-A — Nested Temporal MAG-Gate Probe (任务书 2026-08-14, Step 2).

Question: 当前冻结基线 p 的逐样本相对幅度能否被 market 幅度预测 m̂ 条件化,
且该条件化在 NESTED temporal 协议下存活?

  p  = canonical clean-baseline-v2 OOF (months 21-70, per-block refit,
       source_train_end < month 保证)
  m̂  = MarketSeqNet |y|-head OOF — 与 P5-02I P_mag probe 完全同构
       (同一架构/seed/epochs/cosine loss), 仅在 months 21-40 训练,
       预测 41-50 + 51-70 (严格 OOF)

协议 (嵌套 temporal, 硬约束 §3.2):
  outer block 51-60: gate 只在 months 41-50 拟合 (m̂/p/y 均 OOF), 冻结后推断 51-60
  outer block 61-70: gate 只在 months 41-60 拟合, 冻结后推断 61-70
  bin 分位数阈值只在 gate-fit 窗口内计算 (无 eval 泄漏)
  非嵌套参照 (fit 41-50 → eval 51-70) 仅作诊断, 明确标注, 不作判定依据

Gate Model A: 4 quantile bins of m̂, a_k ∈ [0.5, 2.0], L2(a-1) λ=0.1
Gate Model B: softplus(a + b·rank(m̂)) — 仅当 A 出现稳定信号才运行
Control:      m̂ 在 gate 窗口内置换 (解耦 m̂ 与 p,y) 后同流程 — 检验增益是否
              magnitude 特异, 而非 4 参数对 p 的通用形状优化

判定 (§4.7):
  CONTINUE:  Δcosine_outer ≥ +0.0007 且 ≥70% outer months Δ>0 且 late(61-70)≥0
             且去掉 top-1 gain month 后仍 >0 且 gate std>0 且无 norm 爆炸
  STRONG LIVE: Δ ≥ +0.001 且上述全部满足
  KILL:      Δ < +0.0004 或 gain 集中于 1-2 月 或 late<0 或 gate 退化常数
  INCONCLUSIVE: 其余边界情况 (禁止 threshold hunting)
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
import scipy.optimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

MARKET = Path(r"D:\mscapital-forecasting\data\raw\train\market.feather")
LABEL = Path(r"D:\mscapital-forecasting\data\raw\train\label.feather")
CANON = Path(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
OUT = Path(r"D:\mscapital-kaggle\output\p5a_mag_gate")
OUT.mkdir(parents=True, exist_ok=True)

STEPS = 200
CHANNELS = 18
EPOCHS = 15
BATCH = 1024
LR = 1e-3
WD = 1e-4
SEED = 42
LAM = 0.1          # L2(a-1) 正则强度 (报告说明: 锚定 a≈1, 防过拟合)
ABOUND = (0.5, 2.0)  # 参数范围依据: canonical 各月 pred norm 比值 ~0.6-1.5


def cosine(p: np.ndarray, y: np.ndarray) -> float:
    """Uncentered cosine similarity (global, scale-invariant)."""
    num = float(np.dot(p, y))
    den = float(np.sqrt(np.dot(p, p) * np.dot(y, y)))
    return num / den if den > 0 else 0.0


# ---------------- P5-01/P5-02I exact replica components ----------------

def build_sequence(market_path: Path, sample_ids: np.ndarray, steps: int = STEPS) -> np.ndarray:
    """P5-01 exact replica: 18 channels, uniform 3s grid (0..597s)."""
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
        h = self.embed(x.permute(0, 2, 1))
        h = self.blocks(h)
        return self.head(h.mean(dim=2))


def train_mag(Xtr: np.ndarray, ytr: np.ndarray, device: torch.device) -> nn.Module:
    """P_mag probe exact replica: cosine loss on |y|, 15ep, AdamW 1e-3."""
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
            print(f"    [mag ep {ep:02d}/{EPOCHS}] loss={ep_loss / nb:.5f}", flush=True)
    return model


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), BATCH * 4):
        xb = torch.from_numpy(X[i:i + BATCH * 4]).float().to(device)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs).reshape(-1)


# ---------------- gate models ----------------

def fit_gate4(p: np.ndarray, m: np.ndarray, y: np.ndarray, lam: float = LAM):
    """Model A: 4 quantile bins of m̂, per-bin scale a_k, L2(a-1) reg, bounds."""
    thr = np.quantile(m, [0.25, 0.5, 0.75])
    b = np.digitize(m, thr).astype(int)

    def obj(a):
        pg = a[b] * p
        return -cosine(pg, y) + lam * float(np.sum((a - 1.0) ** 2))

    res = scipy.optimize.minimize(obj, np.ones(4), method="L-BFGS-B",
                                  bounds=[ABOUND] * 4,
                                  options={"maxiter": 500})
    return res.x, thr, res.fun


def apply_gate4(p: np.ndarray, m: np.ndarray, a: np.ndarray, thr: np.ndarray) -> np.ndarray:
    b = np.digitize(m, thr).astype(int)
    return a[b] * p


def fit_gate_softplus(p: np.ndarray, m: np.ndarray, y: np.ndarray):
    """Model B: g = softplus(a + b·rank(m̂)); rank via gate-window CDF."""
    sm = np.sort(m)
    r = np.searchsorted(sm, m) / len(m)

    def obj(ab):
        g = np.log1p(np.exp(np.clip(ab[0] + ab[1] * r, -20, 20)))
        pg = g * p
        return -cosine(pg, y) + LAM * float((ab[0] ** 2 + ab[1] ** 2))

    res = scipy.optimize.minimize(obj, np.array([0.0, 0.0]), method="L-BFGS-B",
                                  bounds=[(-2.0, 3.0), (-4.0, 4.0)],
                                  options={"maxiter": 500})
    return res.x


def apply_gate_softplus(p: np.ndarray, m: np.ndarray, sm: np.ndarray, ab: np.ndarray) -> np.ndarray:
    r = np.searchsorted(sm, m) / len(sm)
    g = np.log1p(np.exp(np.clip(ab[0] + ab[1] * r, -20, 20)))
    return g * p


# ---------------- metrics ----------------

def monthly_table(p_base: np.ndarray, p_new: np.ndarray, y: np.ndarray,
                  month: np.ndarray, months: np.ndarray) -> list[dict]:
    rows = []
    for mm in months:
        mask = month == mm
        if mask.sum() < 50:
            continue
        c0 = cosine(p_base[mask], y[mask])
        c1 = cosine(p_new[mask], y[mask])
        rows.append({
            "month": int(mm), "n": int(mask.sum()),
            "mean_abs_y": float(np.abs(y[mask]).mean()),
            "cos_base": c0, "cos_new": c1, "delta": c1 - c0,
            "std_base": float(p_base[mask].std()), "std_new": float(p_new[mask].std()),
            "norm_ratio": float(p_new[mask].std() / p_base[mask].std()),
        })
    return rows


def decile_decomp(p_base: np.ndarray, p_new: np.ndarray, y: np.ndarray,
                  score: np.ndarray, label: str) -> dict:
    """Cosine gain + inner-product share by score decile."""
    out = {}
    q = np.quantile(score, np.arange(0.1, 1.0, 0.1))
    for i, lo_ in enumerate([-np.inf, *q]):
        hi_ = q[i] if i < 9 else np.inf
        mask = (score >= lo_) & (score <= hi_)
        if mask.sum() < 100:
            continue
        out[f"d{i+1}"] = {
            "n": int(mask.sum()),
            "delta": float(cosine(p_new[mask], y[mask]) - cosine(p_base[mask], y[mask])),
            "share_base": float((p_base[mask] * y[mask]).sum()) / float((p_base * y).sum()),
            "share_new": float((p_new[mask] * y[mask]).sum()) / float((p_new * y).sum()),
        }
    return {label: out}


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

    # ---- p: canonical baseline OOF (align to label order) ----
    canon = np.load(CANON)
    cids = canon["sample_id"]
    p_sel = canon["baseline_oof"][np.searchsorted(cids, sid_all[sel_mask])]
    p_fr = canon["baseline_oof"][np.searchsorted(cids, sid_all[fr_mask])]
    y_sel, y_fr = y_all[sel_mask], y_all[fr_mask]
    assert np.isfinite(p_sel).all() and np.isfinite(p_fr).all(), "canonical coverage broken"

    # ---- activity (market snapshot count) ----
    act = pl.read_ipc(MARKET, columns=["sample_id"]).group_by("sample_id").len().sort("sample_id")
    a_ids = act["sample_id"].to_numpy()
    a_cnt = act["len"].to_numpy().astype(np.float64)
    act_fr = a_cnt[np.searchsorted(a_ids, sid_all[fr_mask])]

    # ---- m̂: magnitude OOF (P_mag exact replica) ----
    print("building sequences (21-40)...", flush=True)
    X_tr = build_sequence(MARKET, sid_all[tr_mask])
    mu = X_tr.reshape(-1, CHANNELS).mean(axis=0, keepdims=True).astype(np.float32)
    sd = X_tr.reshape(-1, CHANNELS).std(axis=0, keepdims=True).astype(np.float32) + 1e-6
    X_tr_s = (X_tr - mu) / sd
    model = train_mag(X_tr_s, np.abs(y_all[tr_mask]), device)
    del X_tr, X_tr_s
    gc.collect()
    print("building sequences (41-70)...", flush=True)
    X_va = build_sequence(MARKET, sid_all[va_mask])
    X_va_s = (X_va - mu) / sd
    m_va = predict(model, X_va_s, device)
    del X_va, X_va_s, model
    gc.collect()
    m_sel = m_va[sel_mask[va_mask]]
    m_fr = m_va[fr_mask[va_mask]]

    # reproduction check vs P5-02I P_mag (expect ~0.466 / ~0.427)
    c_sel = float(np.corrcoef(m_sel, np.abs(y_sel))[0, 1])
    c_fr = float(np.corrcoef(m_fr, np.abs(y_fr))[0, 1])
    print(f"m̂ reproduction: corr(m̂,|y|) 41-50={c_sel:+.4f} (expect ~0.466) "
          f"51-70={c_fr:+.4f} (expect ~0.427)", flush=True)

    results = {
        "device": str(device),
        "mhat_repro": {"41-50": c_sel, "51-70": c_fr},
        "protocol": "nested temporal: gate fit on prior months only",
        "gateA": {"lam": LAM, "bounds": list(ABOUND)},
        "outer": {}, "diagnostics": {},
    }

    months_outer = np.arange(51, 71)

    def run_outer_block(gw_mask_sel, gw_mask_fr, eval_mask_fr, label):
        """Gate fit window = months < eval block start (using sel+fr windows)."""
        p_gw = np.concatenate([p_sel[gw_mask_sel], p_fr[gw_mask_fr]]) if gw_mask_fr is not None else p_sel[gw_mask_sel]
        m_gw = np.concatenate([m_sel[gw_mask_sel], m_fr[gw_mask_fr]]) if gw_mask_fr is not None else m_sel[gw_mask_sel]
        y_gw = np.concatenate([y_sel[gw_mask_sel], y_fr[gw_mask_fr]]) if gw_mask_fr is not None else y_sel[gw_mask_sel]
        a, thr, _ = fit_gate4(p_gw, m_gw, y_gw)
        p_e = p_fr[eval_mask_fr]
        m_e = m_fr[eval_mask_fr]
        y_e = y_fr[eval_mask_fr]
        p_new = apply_gate4(p_e, m_e, a, thr)
        return p_new, a, thr

    # ---- outer block 51-60: gate fit on 41-50 only ----
    sel_gw = np.ones(len(p_sel), dtype=bool)          # months 41-50 (all)
    fr_gw1 = np.zeros(len(p_fr), dtype=bool)          # no fr months
    ev1 = (m_all[fr_mask] <= 60)
    p_new1, a1, thr1 = run_outer_block(sel_gw, fr_gw1, ev1, "B51_60")
    # ---- outer block 61-70: gate fit on 41-60 ----
    fr_gw2 = (m_all[fr_mask] <= 60)
    ev2 = ~ev1
    p_new2, a2, thr2 = run_outer_block(sel_gw, fr_gw2, ev2, "B61_70")

    p_nested = np.concatenate([p_new1, p_new2])
    y_nested = np.concatenate([y_fr[ev1], y_fr[ev2]])
    p_base_n = np.concatenate([p_fr[ev1], p_fr[ev2]])
    m_nested = np.concatenate([m_fr[ev1], m_fr[ev2]])
    month_nested = np.concatenate([m_all[fr_mask][ev1], m_all[fr_mask][ev2]])

    # ---- control: permuted m̂ (decouple m̂ from p,y in gate window) ----
    rng = np.random.default_rng(SEED + 1)
    m_perm = np.concatenate([m_sel, m_fr])[rng.permutation(len(m_sel) + len(m_fr))]
    m_perm_sel = m_perm[:len(m_sel)]
    m_perm_fr = m_perm[len(m_sel):]
    a_p1, thr_p1, _ = fit_gate4(p_sel, m_perm_sel, y_sel)
    p_perm1 = apply_gate4(p_fr[ev1], m_fr[ev1], a_p1, thr_p1)
    m_gw_perm2 = np.concatenate([m_perm_sel, m_perm_fr[fr_gw2]])
    a_p2, thr_p2, _ = fit_gate4(np.concatenate([p_sel, p_fr[fr_gw2]]), m_gw_perm2,
                                np.concatenate([y_sel, y_fr[fr_gw2]]))
    p_perm2 = apply_gate4(p_fr[ev2], m_fr[ev2], a_p2, thr_p2)
    p_perm = np.concatenate([p_perm1, p_perm2])

    # ---- non-nested reference (diagnostic only) ----
    a_nn, thr_nn, _ = fit_gate4(p_sel, m_sel, y_sel)
    p_nonnested = apply_gate4(p_fr, m_fr, a_nn, thr_nn)

    # ---- metrics ----
    c_base = cosine(p_base_n, y_nested)
    c_new = cosine(p_nested, y_nested)
    c_perm = cosine(p_perm, y_nested)
    c_nn = cosine(p_nonnested, y_fr)
    c_base_fr = cosine(p_fr, y_fr)

    mt = monthly_table(p_base_n, p_nested, y_nested, month_nested, months_outer)
    d_outer = c_new - c_base
    d_late = cosine(p_nested[month_nested >= 61], y_nested[month_nested >= 61]) - \
             cosine(p_base_n[month_nested >= 61], y_nested[month_nested >= 61])
    pos_months = sum(1 for r in mt if r["delta"] > 0)
    n_months = len(mt)

    # top-1 / top-2 gain month removal
    dts = np.array([r["delta"] for r in mt])
    rem1 = np.argsort(dts)[::-1][:1]
    rem2 = np.argsort(dts)[::-1][:2]
    keep1 = ~np.isin(month_nested, np.array([mt[i]["month"] for i in rem1]))
    keep2 = ~np.isin(month_nested, np.array([mt[i]["month"] for i in rem2]))
    d_rem1 = cosine(p_nested[keep1], y_nested[keep1]) - cosine(p_base_n[keep1], y_nested[keep1])
    d_rem2 = cosine(p_nested[keep2], y_nested[keep2]) - cosine(p_base_n[keep2], y_nested[keep2])

    # gate stats
    a_all = np.concatenate([a1, a2])
    gate_stats = {
        "a1": a1.tolist(), "a2": a2.tolist(),
        "mean": float(a_all.mean()), "std": float(a_all.std()),
        "range": [float(a_all.min()), float(a_all.max())],
    }

    # norm check
    norm_ratios = np.array([r["norm_ratio"] for r in mt])
    norm_max = float(norm_ratios.max()) if len(norm_ratios) else float("nan")

    # decompositions
    dec_y = decile_decomp(p_base_n, p_nested, y_nested, np.abs(y_nested), "by_abs_y")
    dec_m = decile_decomp(p_base_n, p_nested, y_nested, m_nested, "by_mhat")

    # activity lo/mid/hi (P5-02I style)
    act_lo = (act_fr >= np.quantile(act_fr, 0.0)) & (act_fr <= np.quantile(act_fr, 0.5))
    act_hi = act_fr > np.quantile(act_fr, 0.9)
    d_act_lo = cosine(p_nested[act_lo], y_nested[act_lo]) - cosine(p_base_n[act_lo], y_nested[act_lo])
    d_act_hi = cosine(p_nested[act_hi], y_nested[act_hi]) - cosine(p_base_n[act_hi], y_nested[act_hi])

    corr_gp = float(np.corrcoef(p_nested, p_base_n)[0, 1])

    results["outer"] = {
        "cos_base_51_70": c_base,
        "cos_gated_51_70": c_new,
        "delta_outer": d_outer,
        "delta_late_61_70": d_late,
        "pos_months": pos_months, "n_months": n_months,
        "delta_after_top1_removal": d_rem1,
        "delta_after_top2_removal": d_rem2,
        "gate_stats": gate_stats,
        "norm_ratio_max": norm_max,
        "corr_gated_base": corr_gp,
        "activity_lo_delta": d_act_lo, "activity_hi_delta": d_act_hi,
        "control_permuted_mhat_delta": c_perm - c_base,
        "nonnested_reference_delta_51_70": c_nn - c_base_fr,
        "decomp": {**dec_y, **dec_m},
        "monthly": mt,
    }

    # ---- Model B (softplus) — only if Model A shows stable signal ----
    dA = d_outer
    if dA >= 0.0004 and pos_months / max(n_months, 1) >= 0.6 and d_late >= 0:
        print("Model A stable → running Model B (softplus monotone gate)", flush=True)
        # block 51-60 fit on 41-50
        ab1 = fit_gate_softplus(p_sel, m_sel, y_sel)
        p_b1 = apply_gate_softplus(p_fr[ev1], m_fr[ev1], np.sort(m_sel), ab1)
        # block 61-70 fit on 41-60
        m_gw2 = np.concatenate([m_sel, m_fr[fr_gw2]])
        p_gw2 = np.concatenate([p_sel, p_fr[fr_gw2]])
        y_gw2 = np.concatenate([y_sel, y_fr[fr_gw2]])
        ab2 = fit_gate_softplus(p_gw2, m_gw2, y_gw2)
        p_b2 = apply_gate_softplus(p_fr[ev2], m_fr[ev2], np.sort(m_gw2), ab2)
        p_b = np.concatenate([p_b1, p_b2])
        d_b = cosine(p_b, y_nested) - c_base
        results["outer"]["modelB"] = {
            "delta_outer": d_b,
            "ab1": ab1.tolist(), "ab2": ab2.tolist(),
            "monthly": monthly_table(p_base_n, p_b, y_nested, month_nested, months_outer),
        }
    else:
        results["outer"]["modelB"] = {"skipped": "Model A not stable enough"}

    # ---- decision (§4.7) ----
    crit = {
        "delta_outer >= +0.0007": d_outer >= 0.0007,
        ">=70% months positive": pos_months / max(n_months, 1) >= 0.70,
        "late (61-70) >= 0": d_late >= 0,
        "survives top-1 removal": d_rem1 > 0,
        "gate not constant": gate_stats["std"] > 1e-6,
        "no norm explosion": norm_max < 3.0,
    }
    passed = sum(crit.values())
    if d_outer >= 0.001 and passed == len(crit):
        decision = "STRONG LIVE"
    elif d_outer >= 0.0007 and passed == len(crit):
        decision = "LIVE"
    elif d_outer < 0.0004 or (d_outer < 0.0007 and (d_late < 0 or d_rem1 <= 0)):
        decision = "KILL"
    else:
        decision = "INCONCLUSIVE"
    results["decision"] = decision
    results["criteria"] = crit
    results["criteria_passed"] = f"{passed}/{len(crit)}"

    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    np.savez(OUT / "p5a_preds.npz",
             p_base=p_base_n, p_gated=p_nested, p_perm=p_perm, p_nonnested=p_nonnested,
             mhat=m_nested, y=y_nested, month=month_nested,
             a1=a1, thr1=thr1, a2=a2, thr2=thr2)

    # ---- console summary ----
    print("\n" + "=" * 78)
    print(f"P5-A NESTED MAG-GATE: Δ_outer={d_outer:+.6f}  Δ_late={d_late:+.6f}  "
          f"months={pos_months}/{n_months}  Δ(top1-removed)={d_rem1:+.6f}  "
          f"gate_std={gate_stats['std']:.4f}  norm_max={norm_max:.2f}")
    print(f"  control(permuted m̂)={c_perm - c_base:+.6f}  nonnested_ref={c_nn - c_base_fr:+.6f}  "
          f"corr(gated,base)={corr_gp:.4f}")
    print(f"  a1={np.round(a1,3)} thr1={np.round(thr1,4)} | a2={np.round(a2,3)} thr2={np.round(thr2,4)}")
    print(f"  activity lo={d_act_lo:+.6f} hi={d_act_hi:+.6f}")
    print(f"  decision: {decision}  ({passed}/{len(crit)} criteria)")
    print("=" * 78)
    print(f"\ntotal {time.time()-t0:.0f}s → {OUT}")


if __name__ == "__main__":
    main()
