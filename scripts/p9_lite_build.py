# -*- coding: utf-8 -*-
"""P9-Lite 三探针特征构建 (2026-08-20).

三个包, 全部在 152 基线之上追加新列 (只改变 feature information, 不动协议):

  P9-A  Cancel Pressure   : 按 side 拆的撤单成交量/计数 + 撤单压力 + 相对剩余深度 + 触价撤单 (near-touch by BBO)
  P9-B  Event-Time        : inter-arrival 分布形态 (cv/log) + burst + event acceleration (recent/prev) + 大单尾部
  P9-C  M55-lite          : L1/L2 depth imbalance + 深度加权 DWI + trade direction entropy + DWI*entropy

泄漏纪律 (P0): 全部特征只用本样本 60s 窗口内数据; 无跨样本统计; 无 target 参与.
near-touch 的 BBO 快照取事件同一秒的 market 快照 (滞后 0s, 无未来信息).

来源: raw_ot_agg.parquet (P10-FM 已建 per-sample O/T 聚合) + order.feather (P9-A side-split)
      + market.feather (P9-C DWI) + f0726_train.parquet (基线). 
注意: raw_ot_agg 的 ob_*/tb_* 已含 cancel_side_imb / iat_cv / burst 等原子;
      Z (152+73Z) 是这些原子的条件创新版且已 GREEN —— 本探针测"原始聚合"在裸 152 上的增量 (归因).

用法: .venv/Scripts/python.exe scripts/p9_lite_build.py [--smoke]
输出: output/p9_lite/{pkg}/train_aug.parquet  (sample_id + target + 152基线 + pkg新列)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import polars as pl

RAW = r"D:\mscapital-forecasting\data\raw\train"
BASELINE = r"D:\mscapital-forecasting\data\processed\f0726_train.parquet"
RAWAGG = r"D:\mscapital-kaggle\output\p5b_scfi\raw_ot_agg.parquet"
M1L2 = r"D:\mscapital-kaggle\output\p10_feature_mining\market\features_m1_l2.parquet"
OUT = r"D:\mscapital-kaggle\output\p9_lite"
N_CHUNKS = 8


def load_baseline():
    df = pl.read_parquet(BASELINE)
    assert "target" in df.columns and "sample_id" in df.columns
    feat_cols = [c for c in df.columns if c not in ("sample_id", "target", "month")]
    assert len(feat_cols) == 152, f"基线特征数异常: {len(feat_cols)}"
    return df


# ============================================================ P9-A: Cancel
def build_cancel(smoke: bool) -> pl.DataFrame:
    """按 side 拆撤单成交量/计数 + 压力 + 相对深度 + 触价撤单 (order.feather + market BBO)."""
    t0 = time.time()
    bounds = pl.scan_ipc(f"{RAW}/order.feather").select(
        pl.col("sample_id").min().alias("lo"), pl.col("sample_id").max().alias("hi")).collect()
    lo_all, hi_all = int(bounds["lo"][0]), int(bounds["hi"][0])
    step = max(1, (hi_all - lo_all + 1) // N_CHUNKS)
    ranges = [(lo_all + i * step, lo_all + (i + 1) * step - 1) for i in range(N_CHUNKS)]
    ranges[-1] = (ranges[-1][0], hi_all)
    if smoke:
        ranges = [(lo_all, lo_all + 3000)]

    out = []
    for i, (a, b) in enumerate(ranges):
        t1 = time.time()
        # order 窗口 (最近 60s), 只取必要列
        ord = (pl.scan_ipc(f"{RAW}/order.feather")
               .filter(pl.col("sample_id").is_between(a, b),
                       pl.col("seconds_before_predict") >= -60.0)
               .with_columns(pl.col("seconds_before_predict").floor().cast(pl.Int32).alias("sec"))
               .select(["sample_id", "sec", "price", "volume", "side", "order_action"])
               .collect())
        # market 同块, 仅 60s 窗口的 BBO 快照 (同一秒)
        mkt = (pl.scan_ipc(f"{RAW}/market.feather")
               .filter(pl.col("sample_id").is_between(a, b),
                       pl.col("seconds_before_predict") >= -60.0)
               .with_columns(pl.col("seconds_before_predict").floor().cast(pl.Int32).alias("sec"))
               .select(["sample_id", "sec", "bid_price_1", "ask_price_1"])
               .collect())
        # 同秒多快照取最后一条 (t=0 端)
        mkt = mkt.sort(["sample_id", "sec"]).group_by(["sample_id", "sec"], maintain_order=True).last()
        j = ord.join(mkt, on=["sample_id", "sec"], how="left")
        # 触价: price 在当前秒 BBO 范围内 (tol = 0.5*spread, 防极小 spread; BBO 缺失 → 不判触价)
        j = j.with_columns(
            ((0.5 * (pl.col("ask_price_1") - pl.col("bid_price_1")).abs().fill_null(0.0) + 1e-6)).alias("tol"))
        j = j.with_columns(
            ((pl.col("price") >= pl.col("bid_price_1").fill_null(0.0) - pl.col("tol")) &
             (pl.col("price") <= pl.col("ask_price_1").fill_null(0.0) + pl.col("tol")))
            .fill_null(False).alias("near_touch"),
            (pl.col("sec") >= -15).alias("near15"),
            (pl.col("side") == 0).alias("is_bid"),
            (pl.col("order_action") == 1).alias("is_cancel"))
        j = j.drop("tol")
        g = j.group_by("sample_id").agg(
            ca_vol=pl.col("volume").filter(pl.col("is_cancel") & ~pl.col("is_bid")).sum(),
            cb_vol=pl.col("volume").filter(pl.col("is_cancel") & pl.col("is_bid")).sum(),
            ca_cnt=pl.col("volume").filter(pl.col("is_cancel") & ~pl.col("is_bid")).count(),
            cb_cnt=pl.col("volume").filter(pl.col("is_cancel") & pl.col("is_bid")).count(),
            ca_vol_near=pl.col("volume").filter(pl.col("is_cancel") & ~pl.col("is_bid") & pl.col("near15")).sum(),
            cb_vol_near=pl.col("volume").filter(pl.col("is_cancel") & pl.col("is_bid") & pl.col("near15")).sum(),
            ca_cnt_near=pl.col("volume").filter(pl.col("is_cancel") & ~pl.col("is_bid") & pl.col("near15")).count(),
            cb_cnt_near=pl.col("volume").filter(pl.col("is_cancel") & pl.col("is_bid") & pl.col("near15")).count(),
            nt_ca_vol=pl.col("volume").filter(pl.col("is_cancel") & ~pl.col("is_bid") & pl.col("near_touch")).sum(),
            nt_cb_vol=pl.col("volume").filter(pl.col("is_cancel") & pl.col("is_bid") & pl.col("near_touch")).sum(),
            nt_ca_cnt=pl.col("volume").filter(pl.col("is_cancel") & ~pl.col("is_bid") & pl.col("near_touch")).count(),
            nt_cb_cnt=pl.col("volume").filter(pl.col("is_cancel") & pl.col("is_bid") & pl.col("near_touch")).count(),
        )
        out.append(g)
        print(f"  cancel chunk[{i+1}/{len(ranges)}]: {time.time()-t1:.0f}s samples={g.height:,} rows={ord.height:,}",
              flush=True)

    f = pl.concat(out).sort("sample_id").with_columns(pl.all().fill_null(0.0))
    eps = 1e-9
    f = f.with_columns([
        ((pl.col("cb_vol") - pl.col("ca_vol")) / (pl.col("cb_vol") + pl.col("ca_vol") + eps)).alias("cancel_press_vol"),
        ((pl.col("cb_cnt") - pl.col("ca_cnt")) / (pl.col("cb_cnt") + pl.col("ca_cnt") + eps)).alias("cancel_press_cnt"),
        ((pl.col("cb_vol_near") - pl.col("ca_vol_near")) / (pl.col("cb_vol_near") + pl.col("ca_vol_near") + eps)).alias("cancel_press_vol_near"),
        ((pl.col("cb_cnt_near") - pl.col("ca_cnt_near")) / (pl.col("cb_cnt_near") + pl.col("ca_cnt_near") + eps)).alias("cancel_press_cnt_near"),
        ((pl.col("nt_cb_vol") - pl.col("nt_ca_vol")) / (pl.col("nt_cb_vol") + pl.col("nt_ca_vol") + eps)).alias("nt_cancel_press_vol"),
        ((pl.col("nt_cb_cnt") - pl.col("nt_ca_cnt")) / (pl.col("nt_cb_cnt") + pl.col("nt_ca_cnt") + eps)).alias("nt_cancel_press_cnt"),
        (pl.col("ca_vol_near") / (pl.col("ca_vol") + pl.col("ca_vol_near") + eps)).alias("cancel_ask_recency"),
        (pl.col("cb_vol_near") / (pl.col("cb_vol") + pl.col("cb_vol_near") + eps)).alias("cancel_bid_recency"),
        ((pl.col("nt_cb_vol") + pl.col("nt_ca_vol")) / (pl.col("cb_vol") + pl.col("ca_vol") + eps)).alias("cancel_ntouch_share"),
    ])
    f = f.with_columns(pl.col("sample_id").cast(pl.Int32))
    print(f"[P9-A] cancel build done ({time.time()-t0:.0f}s) cols={f.width}", flush=True)
    return f


# ============================================================ P9-B: Event-Time
def build_eventtime() -> pl.DataFrame:
    """从 raw_ot_agg 挑 iat 分布形态 / burst / acceleration / 大单尾部 (不在 152 的原子)."""
    a = pl.read_parquet(RAWAGG).sort("sample_id")
    cols = ["sample_id",
            "ob_iat_cv", "ob_iat_mean_log", "ob_burst_ratio",
            "ob_recent_prev_rate", "ob_recent_prev_vol",
            "ob_recent15_sv", "ob_prev30_sv", "ob_size_p90", "ob_size_max",
            "tb_iat_cv", "tb_iat_mean_log", "tb_burst_ratio",
            "tb_recent_prev_rate", "tb_recent_prev_vol",
            "tb_recent15_sv", "tb_prev30_sv", "tb_size_p90", "tb_size_max",
            ]
    f = a.select(cols).with_columns(pl.all().fill_null(0.0))
    f = f.with_columns(pl.col("sample_id").cast(pl.Int32))
    return f


# ============================================================ P9-C: M55-lite
def build_m55(smoke: bool) -> pl.DataFrame:
    """L1/L2 深度加权 DWI (market 60s) + L2 imbalance 结构 + trade entropy + DWI*entropy."""
    t0 = time.time()
    # --- DWI: 逐秒秒级 dwi_t = (b1-a1 + 2(b2-a2))/(b1+a1+2(b2+a2)) ---
    bounds = pl.scan_ipc(f"{RAW}/market.feather").select(
        pl.col("sample_id").min().alias("lo"), pl.col("sample_id").max().alias("hi")).collect()
    lo_all, hi_all = int(bounds["lo"][0]), int(bounds["hi"][0])
    step = max(1, (hi_all - lo_all + 1) // N_CHUNKS)
    ranges = [(lo_all + i * step, lo_all + (i + 1) * step - 1) for i in range(N_CHUNKS)]
    ranges[-1] = (ranges[-1][0], hi_all)
    if smoke:
        ranges = [(lo_all, lo_all + 3000)]

    dwi_out = []
    for i, (a, b) in enumerate(ranges):
        t1 = time.time()
        m = (pl.scan_ipc(f"{RAW}/market.feather")
             .filter(pl.col("sample_id").is_between(a, b),
                     pl.col("seconds_before_predict") >= -60.0)
             .with_columns([
                 ((pl.col("bid_volume_1") - pl.col("ask_volume_1") +
                   2.0 * (pl.col("bid_volume_2") - pl.col("ask_volume_2"))) /
                  (pl.col("bid_volume_1") + pl.col("ask_volume_1") +
                   2.0 * (pl.col("bid_volume_2") + pl.col("ask_volume_2")) + 1e-9)).alias("dwi"),
             ])
             .select(["sample_id", "dwi"])
             .collect())
        g = m.group_by("sample_id").agg(
            dwi_last=pl.col("dwi").last(), dwi_mean=pl.col("dwi").mean())
        dwi_out.append(g)
        print(f"  m55 chunk[{i+1}/{len(ranges)}]: {time.time()-t1:.0f}s samples={g.height:,}", flush=True)
    dwi = pl.concat(dwi_out).sort("sample_id")

    # --- entropy: 从 raw_ot_agg 的 tb 买卖量 / 计数 ---
    a = pl.read_parquet(RAWAGG).sort("sample_id")
    m1 = pl.read_parquet(M1L2).sort("sample_id")
    if smoke:
        lo0, hi0 = ranges[0]
        a = a.filter(pl.col("sample_id").is_between(lo0, hi0))
        m1 = m1.filter(pl.col("sample_id").is_between(lo0, hi0))
    a = a.with_columns(pl.col("sample_id").cast(pl.Int32))
    def ent(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return (-(p * np.log2(p) + (1 - p) * np.log2(1 - p))).astype(np.float32)  # [0,1] 已归一 (log2)
    a = a.with_columns([
        (pl.col("tb_buy_vol") / (pl.col("tb_buy_vol") + pl.col("tb_sell_vol") + 1e-9)).alias("_pb_vol"),
        (pl.col("tb_buy_cnt") / (pl.col("tb_buy_cnt") + pl.col("tb_sell_cnt") + 1e-9)).alias("_pb_cnt"),
    ])
    ent_vol = a.select(pl.col("_pb_vol").map_batches(lambda s: pl.Series(ent(s.to_numpy())), return_dtype=pl.Float32)).to_series().rename("trade_ent_vol")
    ent_cnt = a.select(pl.col("_pb_cnt").map_batches(lambda s: pl.Series(ent(s.to_numpy())), return_dtype=pl.Float32)).to_series().rename("trade_ent_cnt")

    m1 = m1.with_columns(pl.col("sample_id").cast(pl.Int32))
    a = a.with_columns(pl.col("sample_id").cast(pl.Int32))
    dwi = dwi.with_columns(pl.col("sample_id").cast(pl.Int32))
    f = (dwi
         .join(m1.select(["sample_id", "l2_imb2_last", "l2_imb2_mean",
                          "l2_imb12_diff_last", "l2_imb12_diff_mean", "l2_dep_ratio_last"]),
               on="sample_id", how="left")
         .join(a.select(["sample_id", "_pb_vol", "_pb_cnt"]), on="sample_id", how="left")
         .with_columns([
             ent_vol, ent_cnt,
             (pl.col("dwi_last") * (1.0 - pl.col("_pb_vol"))).alias("dwi_conf"),
         ])
         .drop(["_pb_vol", "_pb_cnt"])
         .with_columns(pl.all().fill_null(0.0)))
    f = f.with_columns(pl.col("sample_id").cast(pl.Int32))
    print(f"[P9-C] m55 build done ({time.time()-t0:.0f}s) cols={f.width}", flush=True)
    return f


# ============================================================ 组装
PKG_COLS = {
    "A": ["cancel_press_vol", "cancel_press_cnt", "cancel_press_vol_near", "cancel_press_cnt_near",
          "nt_cancel_press_vol", "nt_cancel_press_cnt", "cancel_ask_recency", "cancel_bid_recency",
          "cancel_ntouch_share", "ca_vol", "cb_vol", "ca_cnt", "cb_cnt"],
    "B": ["ob_iat_cv", "ob_iat_mean_log", "ob_burst_ratio", "ob_recent_prev_rate", "ob_recent_prev_vol",
          "ob_recent15_sv", "ob_prev30_sv", "ob_size_p90", "ob_size_max",
          "tb_iat_cv", "tb_iat_mean_log", "tb_burst_ratio", "tb_recent_prev_rate", "tb_recent_prev_vol",
          "tb_recent15_sv", "tb_prev30_sv", "tb_size_p90", "tb_size_max"],
    "C": ["dwi_last", "dwi_mean", "l2_imb2_last", "l2_imb2_mean", "l2_imb12_diff_last",
          "l2_imb12_diff_mean", "l2_dep_ratio_last", "trade_ent_vol", "trade_ent_cnt", "dwi_conf"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pkg", choices=["A", "B", "C", "all"], default="all")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    base = load_baseline()
    print(f"baseline {base.shape} feats={base.width-2}", flush=True)

    builds = {"A": None, "B": None, "C": None}
    if args.pkg in ("A", "all"):
        builds["A"] = build_cancel(args.smoke)
    if args.pkg in ("B", "all"):
        builds["B"] = build_eventtime()
    if args.pkg in ("C", "all"):
        builds["C"] = build_m55(args.smoke)

    for pkg, featdf in builds.items():
        if featdf is None:
            continue
        cols = PKG_COLS[pkg]
        missing = [c for c in cols if c not in featdf.columns]
        assert not missing, f"P9-{pkg} 缺列: {missing}"
        aug = (base.join(featdf.select(["sample_id"] + cols), on="sample_id", how="left")
               .with_columns([pl.col(c).fill_null(0.0).fill_nan(0.0).cast(pl.Float32) for c in cols]))
        assert aug.height == base.height, "join broken"
        outdir = f"{OUT}/pkg{pkg}"
        os.makedirs(outdir, exist_ok=True)
        outp = f"{outdir}/train_aug.parquet"
        aug.write_parquet(outp)
        print(f"[P9-{pkg}] wrote {aug.shape} ({len(cols)} new) -> {outp}", flush=True)


if __name__ == "__main__":
    main()
