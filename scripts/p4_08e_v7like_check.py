# -*- coding: utf-8 -*-
"""P4-08E: fusion-object check - is the OOF +0.009 specific to CleanBaseline?

Hypothesis: P4-08A validated "CleanBaseline + cosine-MLP"; production blends
"v7 + cosine". Different fusion objects -> different gain.

Design:
1. train cosine-MLP on months 21-40 (152 feats)
2. predict months 41-70
3. on the 41-70 subset: build v7_like = 0.8*unit(v5_oof) + 0.2*unit(RL_oof)
   (from rlps pseudo OOF, months 33-70)
4. measure delta vs CleanBaseline AND vs v7_like, same cosine prediction
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from mscapital.metrics import cosine_uncentered, normalize_prediction

OUT = Path(r"D:\mscapital-kaggle\output\p4_08e_v7like_check")
OUT.mkdir(parents=True, exist_ok=True)
EPOCHS, BATCH, LR, SEED = 15, 8192, 1e-3, 42


class MLP(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.SiLU(), nn.Dropout(0.01),
            nn.Linear(256, 256), nn.SiLU(), nn.Dropout(0.01),
            nn.Linear(256, 64), nn.SiLU(), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).reshape(-1)


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    canon = np.load(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
    cids, cmonth = canon["sample_id"], canon["month"].astype(int)
    cy, cbase = canon["target"], canon["baseline_oof"]

    fe = pl.read_parquet(r"D:\mscapital-kaggle\scripts\kaggle_0726ds\f0726_train_f32.parquet").sort("sample_id")
    pos = np.searchsorted(fe["sample_id"].to_numpy(), cids)
    names = [c for c in fe.columns if c not in ("sample_id", "target")]
    X = np.nan_to_num(fe.select(names).to_numpy()[pos].astype(np.float32), nan=0.0)

    tr = (cmonth >= 21) & (cmonth <= 40)
    va = (cmonth >= 41) & (cmonth <= 70)
    mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-8
    Xs = (X - mu) / sd

    model = MLP(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    Xt = torch.from_numpy(Xs[tr]).to(device)
    yt = torch.from_numpy(cy[tr].astype(np.float32)).to(device)
    n = tr.sum()
    print("training cosine-MLP on months 21-40...")
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            p = model(Xt[idx])
            loss = 1.0 - torch.nn.functional.cosine_similarity(
                p.reshape(1, -1), yt[idx].reshape(1, -1), dim=1).squeeze()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        xv = torch.from_numpy(Xs[va]).to(device)
        pv = model(xv).cpu().numpy()
    print(f"predicted {va.sum()} rows (months 41-70)")

    yv = cy[va]
    bv = cbase[va]
    # --- (a) CleanBaseline + cosine ---
    for a in (0.0, 0.13, 0.25):
        b_n, _ = normalize_prediction(bv, "rms")
        p_n, _ = normalize_prediction(pv, "rms")
        s = cosine_uncentered(b_n + a * p_n, yv)
        if a == 0.0:
            base_score = s
        else:
            print(f"(a) CleanBaseline + {a}*cosine: score={s:.6f} delta={s - base_score:+.6f}")

    # --- (b) v7_like + cosine ---
    # rlps OOF: months 33-70, ordered by sample_id (f0726_train sorted)
    rl = np.load(r"D:\mscapital-kaggle\output\rlps_final\realmlp_pseudo_pred.npz")["pred"]
    v5 = np.load(r"D:\mscapital-kaggle\output\rlps_v12\v5_table_pseudo_pred.npz")["pred"]
    fe_all = pl.read_parquet(r"D:\mscapital-forecasting\data\processed\f0726_train_f32.parquet").sort("sample_id")
    lab = pl.read_ipc(r"D:\mscapital-forecasting\data\raw\train\label.feather").sort("sample_id")
    m_all = lab["month"].to_numpy()
    rl_mask = (m_all >= 33) & (m_all <= 70)
    rl_ids = fe_all["sample_id"].to_numpy()[rl_mask]
    rl_pred, v5_pred = rl, v5  # rlps 数组已是 33-70 筛选且按 sample_id 排序
    # align to canonical 41-70 subset
    cids_va = cids[va]
    rp = np.searchsorted(rl_ids, cids_va)
    ok = rp < len(rl_ids)
    rp = rp[ok]
    if not np.array_equal(rl_ids[rp], cids_va[ok]):
        # fallback: intersect by isin
        rl_set = set(rl_ids.tolist())
        keep = np.array([int(s) in rl_set for s in cids_va])
        rp2 = np.searchsorted(rl_ids, cids_va[keep])
        rl_p, v5_p = rl_pred[rp2], v5_pred[rp2]
        yv2, bv2, pv2 = yv[keep], bv[keep], pv[keep]
    else:
        keep, rl_p, v5_p = ok, rl_pred[rp], v5_pred[rp]
        yv2, bv2, pv2 = yv[keep], bv[keep], pv[keep]
    print(f"\nv7-like subset: {keep.sum()} rows (41-70 ∩ 33-70)")
    # 生产 v7 公式 (43_lb142_fusion.py): p_v7 = 0.8*p_v5 + 0.2*p_rl, 直接加权
    v7_like = 0.8 * v5_p + 0.2 * rl_p
    print(f"v7_like: mean={v7_like.mean():.2e} std={v7_like.std():.2e} "
          f"(components v5 std={v5_p.std():.2e} rl std={rl_p.std():.2e})")
    # 关键诊断: cosine 与两个 baseline 的 residual 相关
    r_clean = yv2 - bv2
    r_v7 = yv2 - v7_like
    print(f"\n=== residual attribution (核心) ===")
    print(f"corr(cos, residual_CleanBaseline) = {np.corrcoef(pv2, r_clean)[0,1]:+.4f}")
    print(f"corr(cos, residual_v7)            = {np.corrcoef(pv2, r_v7)[0,1]:+.4f}")
    print(f"corr(cos, v7_like)                = {np.corrcoef(pv2, v7_like)[0,1]:+.4f}")
    print(f"corr(cos, CleanBaseline)          = {np.corrcoef(pv2, bv2)[0,1]:+.4f}")
    # α 网格扫描 (合法: 全部在 41-70 评估, 单 fold)
    print(f"\n=== alpha sweep (v7_like + a*unit(cos)) ===")
    p_n, _ = normalize_prediction(pv2, "rms")
    for a in (0.0, 0.03, 0.05, 0.08, 0.10, 0.13, 0.17, 0.25):
        v7_n, _ = normalize_prediction(v7_like, "rms")
        s = cosine_uncentered(v7_n + a * p_n, yv2)
        if a == 0.0:
            s0 = s
            print(f"  a=0.00: score={s:.6f} (baseline)")
        else:
            print(f"  a={a:.2f}: score={s:.6f} delta={s - s0:+.6f}")
    print(f"\n=== alpha sweep (CleanBaseline + a*unit(cos), 同子集) ===")
    for a in (0.0, 0.13, 0.25):
        b_n, _ = normalize_prediction(bv2, "rms")
        s = cosine_uncentered(b_n + a * p_n, yv2)
        if a == 0.0:
            s0c = s
        else:
            print(f"  a={a:.2f}: score={s:.6f} delta={s - s0c:+.6f}")
    # monthly delta for v7_like + 0.13
    print(f"\n=== monthly delta (v7_like + 0.13*cos) ===")
    v7_n, _ = normalize_prediction(v7_like, "rms")
    fin = v7_n + 0.13 * p_n
    pos_m = 0
    tot_m = 0
    for m in range(41, 71):
        mm = (cmonth[va][keep] == m)
        if mm.sum() > 100:
            d = cosine_uncentered(fin[mm], yv2[mm]) - cosine_uncentered(v7_n[mm], yv2[mm])
            tot_m += 1
            pos_m += d > 0
            if abs(d) > 0.001:
                print(f"  month {m}: {d:+.5f}")
    print(f"months positive: {pos_m}/{tot_m}")
    # 分段: 41-50 (T3/T4 验证过的范围) vs 51-70 (从未验证)
    for lo_, hi_, lbl in ((41, 50, "41-50 (已验证区)"), (51, 60, "51-60 (未验证区)"), (61, 70, "61-70 (未验证区)")):
        mm = (cmonth[va][keep] >= lo_) & (cmonth[va][keep] <= hi_)
        if mm.sum() > 1000:
            d = cosine_uncentered(fin[mm], yv2[mm]) - cosine_uncentered(v7_n[mm], yv2[mm])
            d_c = cosine_uncentered(normalize_prediction(bv2[mm], "rms")[0] + 0.13 * p_n[mm], yv2[mm]) \
                  - cosine_uncentered(normalize_prediction(bv2[mm], "rms")[0], yv2[mm])
            print(f"  {lbl}: v7+cos delta={d:+.6f} | Clean+cos delta={d_c:+.6f}")
    print("\nwritten to", OUT)


if __name__ == "__main__":
    main()
