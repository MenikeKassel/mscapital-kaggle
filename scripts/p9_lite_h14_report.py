# -*- coding: utf-8 -*-
"""H14-V1 验证报告 (v2): 3 seed × {base 152, raw cancel +13, mask cancel +13}.

回答: (1) raw 撤单增益是否多 seed 稳健; (2) cancel-activity mask (低撤单样本置0) 是否如
机制猜想消除 regime 集中 — 同 seed 受控比较 raw vs mask.

判定: 同 seed Δmask − Δraw ≥ 0 且 mask 的 low_act 回正 → 机理猜想成立; 否则证伪.
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

OUT = r"D:\mscapital-kaggle\output\p9_lite\pkgA"
SEEDS = [2026, 2027, 2028]
DIR = {
    "base": {2026: "base", 2027: "base_s2027", 2028: "base_s2028"},
    "raw":  {2026: "feat",  2027: "feat_s2027",  2028: "feat_s2028"},
    "mask": {2026: "feat_mask", 2027: "feat_s2027_mask", 2028: "feat_s2028_mask"},
}


def load(d):
    with open(f"{OUT}/{d}/results.json") as f:
        r = json.load(f)
    return r, np.load(f"{OUT}/{d}/preds.npz")


def cos(p, y, sel):
    p = p[sel].astype(np.float64); y = y[sel].astype(np.float64)
    return float(p @ y / (np.sqrt(p @ p) * np.sqrt(y @ y) + 1e-30))


def main():
    R = {k: {s: load(DIR[k][s]) for s in SEEDS} for k in ("base", "raw", "mask")}

    print("frozen 51-70 by seed:")
    print(f"  {'seed':>6}{'base':>10}{'rawΔ':>10}{'maskΔ':>10}  (arm−base per seed)")
    d_raw, d_mask = [], []
    for s in SEEDS:
        b = R["base"][s][0]["frozen51_70_cosine"]
        r = R["raw"][s][0]["frozen51_70_cosine"]
        m = R["mask"][s][0]["frozen51_70_cosine"]
        d_raw.append(r - b); d_mask.append(m - b)
        print(f"  {s:>6}{b:>10.6f}{r-b:>+10.6f}{m-b:>+10.6f}")
    d_raw, d_mask = np.array(d_raw), np.array(d_mask)
    print(f"\nraw  Δ: mean={d_raw.mean():+.6f} std={d_raw.std():+.6f} min={d_raw.min():+.6f} max={d_raw.max():+.6f}  pos_seed={np.sum(d_raw>0)}/3")
    print(f"mask Δ: mean={d_mask.mean():+.6f} std={d_mask.std():+.6f} min={d_mask.min():+.6f} max={d_mask.max():+.6f}  pos_seed={np.sum(d_mask>0)}/3")
    print(f"mask−raw (同 seed, + 表示 mask 更优): {[(d_mask[i]-d_raw[i]).round(6) for i in range(3)]}")

    # regime 分裂 (seed 2026, 受控最干净)
    b26, r26, m26 = R["base"][2026][1], R["raw"][2026][1], R["mask"][2026][1]
    mm = b26["month"]; fr = (mm >= 51) & (mm <= 70)
    o = (pl.read_parquet(r"D:\mscapital-kaggle\output\p9_lite\pkgA\train_aug.parquet",
                         columns=["sample_id", "o_n_45"]).join(
        pl.DataFrame({"sample_id": b26["sample_id"]}), on="sample_id").to_numpy()[:, 1])
    thr = np.median(o[fr]); lo = (o < thr); hi = ~lo
    print("\nregime 分裂 (seed 2026, frozen, o_n_45 median):")
    for nm, sel in [("low_act", lo), ("hi_act", hi)]:
        rb = cos(b26["pred"], b26["y"], fr & sel)
        rr = cos(r26["pred"], r26["y"], fr & sel)
        rm = cos(m26["pred"], m26["y"], fr & sel)
        print(f"  {nm:8s} base={rb:.6f} raw={rr:.6f} mask={rm:.6f}  Δraw={rr-rb:+.6f}  Δmask={rm-rb:+.6f}")
    c = float(np.corrcoef(r26["pred"][fr], m26["pred"][fr])[0, 1])
    print(f"\nmask(2026) vs raw(2026) pred corr = {c:.4f}")


if __name__ == "__main__":
    main()
