# -*- coding: utf-8 -*-
"""三方对比报告: 152 base vs +cancel vs +Z vs +cancel+Z (frozen 51-70, seed 2026).

回答: (1) cancel 能否替代 Z; (2) cancel 能否在 Z 之上叠加; (3) 相关结构.
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

OUT = r"D:\mscapital-kaggle\output\p9_lite\pkgA"
ARMS = {"base": "base", "cancel": "feat", "Z": "feat_z", "cancel+Z": "feat_j"}


def load(d):
    with open(f"{OUT}/{d}/results.json") as f:
        r = json.load(f)
    return r, np.load(f"{OUT}/{d}/preds.npz")


def cos(p, y, sel):
    p = p[sel].astype(np.float64); y = y[sel].astype(np.float64)
    return float(p @ y / (np.sqrt(p @ p) * np.sqrt(y @ y) + 1e-30))


def main():
    R = {k: load(v) for k, v in ARMS.items()}
    print("frozen 51-70 (seed 2026):")
    frz = {k: v[0]["frozen51_70_cosine"] for k, v in R.items()}
    b = frz["base"]
    for k in ["cancel", "Z", "cancel+Z"]:
        print(f"  {k:9s} {frz[k]:.6f}  Δ={frz[k]-b:+.6f}  pos={R[k][0]['frozen51_70_pos_months']}/20")

    # 替代: cancel vs Z ; 叠加: cancel+Z vs Z
    print(f"\n替代判定 (cancel 能否替 Z):    Δcancel={frz['cancel']-b:+.6f} vs ΔZ={frz['Z']-b:+.6f}")
    print(f"叠加判定 (cancel 能否叠 Z):    Δ(cancel+Z)={frz['cancel+Z']-b:+.6f} vs ΔZ={frz['Z']-b:+.6f} "
          f"→ J−Z={frz['cancel+Z']-frz['Z']:+.6f}")

    # 相关结构 (frozen)
    fr = (R["base"][1]["month"] >= 51)
    print("\npred corr (frozen 51-70):")
    keys = ["base", "cancel", "Z", "cancel+Z"]
    for a in keys:
        for b_ in keys:
            c = float(np.corrcoef(R[a][1]["pred"][fr], R[b_][1]["pred"][fr])[0, 1])
            print(f"  {a:9s} x {b_:9s} = {c:.4f}")

    # 月度 Δ (frozen) 明细: J vs Z
    dZ = {int(k): float(v) for k, v in R["Z"][0]["monthly_cosine"].items()}
    dJ = {int(k): float(v) for k, v in R["cancel+Z"][0]["monthly_cosine"].items()}
    db = {int(k): float(v) for k, v in R["base"][0]["monthly_cosine"].items()}
    diffs = [(m, (dJ.get(m, 0) - db.get(m, 0)) - (dZ.get(m, 0) - db.get(m, 0))) for m in range(51, 71) if m in db]
    pos = sum(1 for _, v in diffs if v > 0)
    print(f"\nJ−Z frozen 月度 Δ: pos={pos}/20")


if __name__ == "__main__":
    main()
