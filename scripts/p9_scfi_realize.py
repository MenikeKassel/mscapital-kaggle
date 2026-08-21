# -*- coding: utf-8 -*-
"""H-SCFI-REALIZE: 兑现 SCFI 条件创新特征 (P5-04/06/07 已验证 +0.0040, 未进生产).

A 臂: 152 特征 (spot_rmlp_A.parquet, 全样本)
C 臂: 152 + 73 Z 特征 (spot_rmlp_C.parquet, 全样本, Z 为 cross-fit 条件创新残差)
骨架: C-05 同协议 (plain 3x256 MLP, robust+clip fold-local, MSE 训练,
      best-epoch global cosine 选择, PSEUDO m0-32/m33-70)
门禁: ΔPSEUDO(C-A) > +0.001 且月度方向稳
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from c05_recipe_e0 import RobustScaleSmoothClip, MLP, cosine_uncentered, set_seed

RAW = Path(r"D:\mscapital-forecasting\data\raw\train")
OUT = Path(r"D:\mscapital-kaggle\output\p9_scfi_realize")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 2026
EPOCHS = 30
BATCH = 512
LR = 1e-3
HIDDEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_arm(name: str, feat_path: Path, out: Path):
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(feat_path)
    lab = pl.read_ipc(RAW / "label.feather").select(["sample_id", "month"])
    df = df.join(lab, on="sample_id", how="left")
    feat_cols = [c for c in df.columns if c not in ("sample_id", "month")]
    X = df.select(feat_cols).to_numpy().astype(np.float32)
    y = df["target"].to_numpy().astype(np.float64) if "target" in df.columns else None
    m = df["month"].to_numpy()

    # join target
    labf = pl.read_ipc(RAW / "label.feather").select(["sample_id", "target"])
    df = df.join(labf, on="sample_id", how="left")
    y = df["target"].to_numpy().astype(np.float64)

    tr = m <= 32
    X_tr, y_tr = X[tr], y[tr]
    X_ev, y_ev = X[~tr], y[~tr]
    print(f"[{name}] train {X_tr.shape} / eval {X_ev.shape} device={DEVICE}")

    pp = RobustScaleSmoothClip().fit(X_tr)
    X_tr = np.nan_to_num(pp.transform(X_tr).astype(np.float32), nan=0.0)
    X_ev = np.nan_to_num(pp.transform(X_ev).astype(np.float32), nan=0.0)

    set_seed(SEED)
    model = MLP(X_tr.shape[1], HIDDEN).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
    lossf = torch.nn.MSELoss()

    Xt = torch.from_numpy(X_tr).to(DEVICE)
    yt = torch.from_numpy(y_tr.astype(np.float32)).to(DEVICE)
    Xe = torch.from_numpy(X_ev).to(DEVICE)
    n = len(X_tr)
    hist = []
    best_cos, best_cos_ep = -9, 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = [model(Xe[i:i + 4096]).cpu().numpy() for i in range(0, len(Xe), 4096)]
            p_ev = np.concatenate(pv)
        cos = cosine_uncentered(p_ev, y_ev)
        hist.append({"epoch": ep + 1, "cosine": cos})
        if cos > best_cos:
            best_cos, best_cos_ep = cos, ep + 1
            torch.save(model.state_dict(), out / "best_cos.pt")
        print(f"[{name}] ep {ep+1:02d} cos={cos:.6f}", flush=True)

    model.load_state_dict(torch.load(out / "best_cos.pt"))
    model.eval()
    with torch.no_grad():
        p_final = np.concatenate([model(Xe[i:i + 4096]).cpu().numpy()
                                  for i in range(0, len(Xe), 4096)])
    # monthly deltas
    m_ev = m[~tr]
    monthly = {}
    for mm in np.unique(m_ev):
        mask = m_ev == mm
        monthly[int(mm)] = cosine_uncentered(p_final[mask], y_ev[mask])
    res = {"arm": name, "n_feat": int(X.shape[1]),
           "best_cosine_epoch": best_cos_ep, "best_cosine": best_cos,
           "final_cosine": cosine_uncentered(p_final, y_ev),
           "monthly": monthly, "runtime_s": round(time.time() - t0, 1)}
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[{name}] DONE best_ep={best_cos_ep} cos={best_cos:.6f} ({time.time()-t0:.0f}s)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "C", "both"], default="both")
    args = ap.parse_args()
    P5B = Path(r"D:\mscapital-kaggle\output\p5b_scfi")
    results = {}
    if args.arm in ("A", "both"):
        results["A"] = run_arm("A_152", P5B / "spot_rmlp_A.parquet", OUT / "arm_A")
    if args.arm in ("C", "both"):
        results["C"] = run_arm("C_152Z", P5B / "spot_rmlp_C.parquet", OUT / "arm_C")
    if "A" in results and "C" in results:
        dA, dC = results["A"]["best_cosine"], results["C"]["best_cosine"]
        delta = dC - dA
        npos = sum(1 for mm in results["C"]["monthly"] if results["C"]["monthly"][mm] > results["A"]["monthly"].get(mm, 0))
        nmon = len(results["C"]["monthly"])
        summary = {"delta_C_minus_A": delta, "monthly_positive": f"{npos}/{nmon}"}
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n=== Δ(C−A) = {delta:+.6f}  (A={dA:.6f} → C={dC:.6f}), 月度正 {npos}/{nmon} ===")


if __name__ == "__main__":
    main()
