# -*- coding: utf-8 -*-
"""P10 tx 特征构建 v2 (pandas 版, 绕开 polars 全量 pivot/rename 怪癖).

H1: 4×15s 分段聚合 (t_seg{B}_{stat})
H2: 前30s/后30s 聚合 (t_h{h}_{stat})
H3: 大单拆分模式 (big_*)
泄漏纪律: 全部特征只用本样本 60s 窗口内数据; 无跨样本统计 (大单阈值是全局常数, 与 target 无关).
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

RAW = r"D:\mscapital-forecasting\data\raw\train\transaction.feather"
OUT = r"D:\mscapital-kaggle\output\p10_feature_mining\transaction"
os.makedirs(OUT, exist_ok=True)


def main():
    t0 = time.time()
    df = pd.read_feather(RAW)
    df = df.sort_values(["sample_id", "seconds_before_predict"]).reset_index(drop=True)
    print(f"loaded {len(df):,} rows ({time.time()-t0:.0f}s)", flush=True)

    big_thr = float(df["volume"].quantile(0.90))
    df["_is_buy"] = (df["side"] == 0).astype(np.int8)
    df["_sv"] = df["volume"] * np.where(df["side"] == 0, 1.0, -1.0)
    df["_sv_per_vol"] = df["_sv"] / (df["volume"] + 1e-9)
    df["_big"] = (df["volume"] >= big_thr).astype(np.int8)
    df["_bin"] = np.clip((df["seconds_before_predict"] // 15.0).astype(np.int8), 0, 3)
    df["_late"] = (df["seconds_before_predict"] < 30.0).astype(np.int8)
    df["_gap"] = df.groupby("sample_id")["seconds_before_predict"].diff().abs()
    print(f"base cols ready ({time.time()-t0:.0f}s)", flush=True)

    # ============ H1: 4×15s 分段聚合 ============
    agg_dict = {
        "cnt": ("volume", "size"), "vol": ("volume", "sum"), "sv": ("_sv", "sum"),
        "buy_r": ("_is_buy", "mean"), "px_mean": ("price", "mean"),
        "px_std": ("price", "std"), "gap_mean": ("_gap", "mean"),
    }
    seg = df.groupby(["sample_id", "_bin"]).agg(**agg_dict).reset_index()
    tot_vol = df.groupby("sample_id")["volume"].sum().rename("_tot_vol")
    seg = seg.join(tot_vol, on="sample_id")
    seg["vol_share"] = seg["vol"] / (seg["_tot_vol"] + 1e-9)
    piv = seg.pivot_table(index="sample_id", columns="_bin",
                          values=["cnt", "vol", "sv", "buy_r", "px_mean", "px_std", "gap_mean", "vol_share"])
    piv.columns = [f"t_seg{b}_{s}" for s, b in piv.columns]
    piv = piv.reset_index()
    print(f"H1 seg: {piv.shape} ({time.time()-t0:.0f}s)", flush=True)

    # ============ H2: 前30s / 后30s ============
    half = df.groupby(["sample_id", "_late"]).agg(**agg_dict).reset_index()
    piv2 = half.pivot_table(index="sample_id", columns="_late",
                            values=["cnt", "vol", "sv", "buy_r", "px_mean", "px_std", "gap_mean"])
    piv2.columns = [f"t_h{h}_{s}" for s, h in piv2.columns]
    piv2 = piv2.reset_index()
    print(f"H2 half: {piv2.shape} ({time.time()-t0:.0f}s)", flush=True)

    # ============ H3: 大单拆分模式 ============
    big = df[df["_big"] == 1]
    big_stats = big.groupby("sample_id").agg(
        big_cnt=("volume", "size"), big_vol=("volume", "sum"),
        big_sgn_mean=("_sv_per_vol", "mean"),
        big_time_pos=("seconds_before_predict", "mean"),
        big_late_share=("seconds_before_predict", lambda x: (x < 15.0).mean()),
        big_max=("volume", "max"), big_mean=("volume", "mean"),
    ).reset_index()
    # run 结构 (连续大单)
    big2 = big.copy()
    big2["_prev_big"] = big2.groupby("sample_id")["_big"].shift(1).fillna(0)
    big2["_run_start"] = ((big2["_big"] == 1) & (big2["_prev_big"] == 0)).astype(np.int8)
    big2["_run_id"] = big2.groupby("sample_id")["_run_start"].cumsum()
    runs = big2.groupby(["sample_id", "_run_id"]).size().rename("run_len").reset_index()
    run_stats = runs.groupby("sample_id").agg(
        big_run_max=("run_len", "max"), big_run_mean=("run_len", "mean"),
        big_n_runs=("run_len", "size"),
    ).reset_index()
    big_gap = big2.groupby("sample_id")["seconds_before_predict"].diff().abs()
    big2["_b_gap"] = big_gap
    big_iat = big2.groupby("sample_id")["_b_gap"].mean().rename("big_iat_mean").reset_index()
    big_stats = (big_stats.merge(run_stats, on="sample_id", how="left")
                          .merge(big_iat, on="sample_id", how="left")
                          .merge(tot_vol.rename("_tot_vol").reset_index(), on="sample_id", how="left"))
    big_stats["big_vol_frac"] = big_stats["big_vol"] / (big_stats["_tot_vol"] + 1e-9)
    tot_cnt = df.groupby("sample_id").size().rename("cnt").reset_index()
    big_stats = big_stats.merge(tot_cnt, on="sample_id", how="left")
    big_stats["big_cnt_frac"] = big_stats["big_cnt"] / (big_stats["cnt"] + 1e-9)
    big_stats = big_stats.drop(columns=["cnt", "_tot_vol"])
    print(f"H3 big: {big_stats.shape} ({time.time()-t0:.0f}s)", flush=True)

    # ============ 合并 ============
    feat = (piv.merge(piv2, on="sample_id", how="outer")
               .merge(big_stats, on="sample_id", how="outer"))
    feat = feat.fillna(0.0)
    feat["sample_id"] = feat["sample_id"].astype(np.int32)
    feat = feat.sort_values("sample_id").reset_index(drop=True)
    out = f"{OUT}/tx_seg_big_features.parquet"
    feat.to_parquet(out, index=False)
    print(f"saved {feat.shape} -> {out} ({time.time()-t0:.0f}s)", flush=True)
    print("cols:", list(feat.columns), flush=True)


if __name__ == "__main__":
    main()
