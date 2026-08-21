# -*- coding: utf-8 -*-
"""H14-V1: Cancel-activity 条件化 mask (2026-08-20).

假设: P9-A 撤单压力 +0.0041 但 regime 集中 (hi_act +0.010 / low_act −0.006), 机制猜想是
"撤单压力比值在撤单量很小的样本上是噪声 → 模型在此过拟合"。本脚本把低撤单样本的
13 个撤单列全置 0 (取消量低于阈值), 让模型在那些样本上完全不用撤单信息。
若成立: mask 变体应 (a) 保住 hi_act 增益, (b) 消除 low_act 拖累 → frozen Δ 保持或更稳.

输出: output/p9_lite/pkgA/train_mask_aug.parquet (与 train_aug.parquet 同 schema, 撤单列被 mask)
阈值: 全样本取消量 (ca_vol+cb_vol) 的 q20.
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl

from p9_lite_build import PKG_COLS

SRC = r"D:\mscapital-kaggle\output\p9_lite\pkgA\train_aug.parquet"
OUT = r"D:\mscapital-kaggle\output\p9_lite\pkgA\train_mask_aug.parquet"
Q = 0.20


def main():
    df = pl.read_parquet(SRC)
    tot = (df["ca_vol"] + df["cb_vol"]).to_numpy()
    thr = float(np.quantile(tot, Q))
    mask = tot >= thr
    print(f"cancel vol: q20={thr:.0f} q10={np.quantile(tot,0.10):.0f} q50={np.quantile(tot,0.50):.0f} "
          f"masked={100*(~mask).mean():.1f}%", flush=True)
    cols = PKG_COLS["A"]
    df2 = df.with_columns([
        pl.Series(c, np.where(mask, df[c].to_numpy(), 0.0)).alias(c) for c in cols
    ])
    df2.write_parquet(OUT)
    print(f"wrote {df2.shape} -> {OUT}", flush=True)
    # 交叉: masked 样本按 order 活动度分布 (验证是否=低活动)
    o = df["o_n_45"].to_numpy()
    print(f"  masked 样本 median o_n_45 = {np.median(o[~mask]):.0f} vs unmasked {np.median(o[mask]):.0f} "
          f"(低取消量确实偏低活动: {np.median(o[~mask]) < np.median(o[mask])})", flush=True)


if __name__ == "__main__":
    main()
