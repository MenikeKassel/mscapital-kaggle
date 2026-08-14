# -*- coding: utf-8 -*-
"""P5-B — Raw Order/Transaction aggregate features (任务书 §5.2 清单).

从原始 event streams 构建新鲜聚合集 (~40 order + ~20 transaction features),
与 f0726 的 153 特征族区分: 本集突出 f0726 缺失的 side×action 拆分
(add/cancel × bid/ask), burstiness (IAT CV / max-gap / 1s 集中度),
size quantiles, recent-vs-previous 强度比。

约定 (项目 27_build_micro_features.py 同源): side 0=buy, 1=sell;
order_action 0=add, 1=cancel。price/volume 已是合成标度, 不做绝对阈值。

输出: output/p5b_scfi/raw_ot_agg.parquet (sample_id + ob_* + tb_*)
"""
from __future__ import annotations

import time
from pathlib import Path

import polars as pl

RAW = Path(r"D:\mscapital-forecasting\data\raw\train")
OUT = Path(r"D:\mscapital-kaggle\output\p5b_scfi")
OUT.mkdir(parents=True, exist_ok=True)

CHUNK = 200_000  # sample_id 分块, 控制内存


def build_order_chunk(lf, lo: int, hi: int) -> pl.DataFrame:
    """Order aggregates for sample_ids in [lo, hi)."""
    df = (lf.filter(pl.col("sample_id").is_between(lo, hi))
          .sort(["sample_id", "seconds_before_predict"])
          .with_columns([
              pl.when(pl.col("side") == 0).then(1.0).otherwise(-1.0).alias("_sgn"),
              pl.when(pl.col("order_action") == 0).then(1.0).otherwise(-1.0).alias("_act"),
              (pl.col("volume") * pl.when(pl.col("side") == 0).then(1.0).otherwise(-1.0)).alias("_sv"),
              (pl.col("volume") * pl.when(pl.col("order_action") == 0).then(1.0).otherwise(-1.0)).alias("_av"),
              pl.col("seconds_before_predict").diff().over("sample_id").abs().alias("_iat"),
          ])
          .group_by("sample_id")
          .agg([
              # --- counts: side × action ---
              pl.col("_sgn").filter(pl.col("order_action") == 0).count().alias("ob_add_cnt"),
              pl.col("_sgn").filter(pl.col("order_action") == 1).count().alias("ob_cancel_cnt"),
              pl.col("_sgn").filter((pl.col("order_action") == 0) & (pl.col("side") == 0)).count().alias("ob_bid_add_cnt"),
              pl.col("_sgn").filter((pl.col("order_action") == 0) & (pl.col("side") == 1)).count().alias("ob_ask_add_cnt"),
              pl.col("_sgn").filter((pl.col("order_action") == 1) & (pl.col("side") == 0)).count().alias("ob_bid_cancel_cnt"),
              pl.col("_sgn").filter((pl.col("order_action") == 1) & (pl.col("side") == 1)).count().alias("ob_ask_cancel_cnt"),
              # --- volumes ---
              pl.col("_sv").sum().alias("ob_signed_vol"),
              pl.col("_av").sum().alias("ob_act_signed_vol"),
              pl.col("volume").filter(pl.col("side") == 0).sum().alias("ob_buy_vol"),
              pl.col("volume").filter(pl.col("side") == 1).sum().alias("ob_sell_vol"),
              pl.col("volume").filter(pl.col("order_action") == 0).sum().alias("ob_add_vol"),
              pl.col("volume").filter(pl.col("order_action") == 1).sum().alias("ob_cancel_vol"),
              # --- imbalances / ratios ---
              (pl.col("_sv").filter(pl.col("order_action") == 0)).sum().alias("ob_add_sv"),
              (pl.col("_sv").filter(pl.col("order_action") == 1)).sum().alias("ob_cancel_sv"),
              # --- arrival / burstiness ---
              pl.len().alias("ob_event_cnt"),
              pl.col("_iat").mean().alias("ob_iat_mean"),
              pl.col("_iat").std().alias("ob_iat_std"),
              pl.col("_iat").max().alias("ob_iat_max"),
              # --- size distribution ---
              pl.col("volume").quantile(0.25).alias("ob_size_p25"),
              pl.col("volume").quantile(0.75).alias("ob_size_p75"),
              pl.col("volume").quantile(0.90).alias("ob_size_p90"),
              pl.col("volume").max().alias("ob_size_max"),
              # --- price distribution ---
              pl.col("price").mean().alias("ob_px_mean"),
              pl.col("price").std().alias("ob_px_std"),
              # --- recent vs previous (last 15s vs 15-45s) ---
              pl.col("_sv").filter(pl.col("seconds_before_predict") >= 45).count().alias("ob_recent15_cnt"),
              pl.col("_sv").filter((pl.col("seconds_before_predict") >= 15) & (pl.col("seconds_before_predict") < 45)).count().alias("ob_prev30_cnt"),
              pl.col("volume").filter(pl.col("seconds_before_predict") >= 45).sum().alias("ob_recent15_vol"),
              pl.col("volume").filter((pl.col("seconds_before_predict") >= 15) & (pl.col("seconds_before_predict") < 45)).sum().alias("ob_prev30_vol"),
              pl.col("_sv").filter(pl.col("seconds_before_predict") >= 45).sum().alias("ob_recent15_sv"),
              pl.col("_sv").filter((pl.col("seconds_before_predict") >= 15) & (pl.col("seconds_before_predict") < 45)).sum().alias("ob_prev30_sv"),
          ])
          .collect())
    return df


def build_trade_chunk(lf, lo: int, hi: int) -> pl.DataFrame:
    df = (lf.filter(pl.col("sample_id").is_between(lo, hi))
          .sort(["sample_id", "seconds_before_predict"])
          .with_columns([
              pl.when(pl.col("side") == 0).then(1.0).otherwise(-1.0).alias("_sgn"),
              (pl.col("volume") * pl.when(pl.col("side") == 0).then(1.0).otherwise(-1.0)).alias("_sv"),
              pl.col("seconds_before_predict").diff().over("sample_id").abs().alias("_iat"),
          ])
          .group_by("sample_id")
          .agg([
              pl.col("_sgn").filter(pl.col("side") == 0).count().alias("tb_buy_cnt"),
              pl.col("_sgn").filter(pl.col("side") == 1).count().alias("tb_sell_cnt"),
              pl.col("_sv").sum().alias("tb_signed_vol"),
              pl.col("volume").filter(pl.col("side") == 0).sum().alias("tb_buy_vol"),
              pl.col("volume").filter(pl.col("side") == 1).sum().alias("tb_sell_vol"),
              pl.len().alias("tb_trade_cnt"),
              pl.col("_iat").mean().alias("tb_iat_mean"),
              pl.col("_iat").std().alias("tb_iat_std"),
              pl.col("_iat").max().alias("tb_iat_max"),
              pl.col("volume").quantile(0.25).alias("tb_size_p25"),
              pl.col("volume").quantile(0.75).alias("tb_size_p75"),
              pl.col("volume").quantile(0.90).alias("tb_size_p90"),
              pl.col("volume").max().alias("tb_size_max"),
              pl.col("price").mean().alias("tb_px_mean"),
              pl.col("price").std().alias("tb_px_std"),
              pl.col("price").max().alias("tb_px_max"),
              pl.col("price").min().alias("tb_px_min"),
              pl.col("_sv").filter(pl.col("seconds_before_predict") >= 45).count().alias("tb_recent15_cnt"),
              pl.col("_sv").filter((pl.col("seconds_before_predict") >= 15) & (pl.col("seconds_before_predict") < 45)).count().alias("tb_prev30_cnt"),
              pl.col("volume").filter(pl.col("seconds_before_predict") >= 45).sum().alias("tb_recent15_vol"),
              pl.col("volume").filter((pl.col("seconds_before_predict") >= 15) & (pl.col("seconds_before_predict") < 45)).sum().alias("tb_prev30_vol"),
              pl.col("_sv").filter(pl.col("seconds_before_predict") >= 45).sum().alias("tb_recent15_sv"),
              pl.col("_sv").filter((pl.col("seconds_before_predict") >= 15) & (pl.col("seconds_before_predict") < 45)).sum().alias("tb_prev30_sv"),
          ])
          .collect())
    return df


def main() -> None:
    t0 = time.time()
    lab = pl.read_ipc(RAW / "label.feather").select("sample_id", "month")
    ids = lab["sample_id"].to_numpy()
    lo, hi = int(ids.min()), int(ids.max())

    olf = pl.scan_ipc(RAW / "order.feather", memory_map=False)
    tlf = pl.scan_ipc(RAW / "transaction.feather", memory_map=False)

    o_parts, t_parts = [], []
    for a in range(lo, hi + 1, CHUNK):
        b = min(a + CHUNK - 1, hi)
        o_parts.append(build_order_chunk(olf, a, b))
        t_parts.append(build_trade_chunk(tlf, a, b))
        print(f"chunk {a:,}-{b:,} done ({time.time()-t0:.0f}s)", flush=True)

    o_all = pl.concat(o_parts).sort("sample_id")
    t_all = pl.concat(t_parts).sort("sample_id")
    lab2 = lab.join(o_all, on="sample_id", how="left").join(t_all, on="sample_id", how="left")
    lab2 = lab2.with_columns([
        (pl.col("ob_add_cnt") / (pl.col("ob_cancel_cnt") + 1.0)).alias("ob_add_cancel_ratio"),
        ((pl.col("ob_bid_add_cnt") - pl.col("ob_ask_add_cnt")) / (pl.col("ob_add_cnt") + 1.0)).alias("ob_add_side_imb"),
        ((pl.col("ob_bid_cancel_cnt") - pl.col("ob_ask_cancel_cnt")) / (pl.col("ob_cancel_cnt") + 1.0)).alias("ob_cancel_side_imb"),
        (pl.col("ob_signed_vol") / (pl.col("ob_buy_vol") + pl.col("ob_sell_vol") + 1.0)).alias("ob_sv_imb"),
        (pl.col("ob_iat_std") / (pl.col("ob_iat_mean") + 1e-6)).alias("ob_iat_cv"),
        (pl.col("ob_iat_max") / (pl.col("ob_iat_mean") + 1e-6)).alias("ob_burst_ratio"),
        ((pl.col("ob_recent15_cnt") / 15.0) / (pl.col("ob_prev30_cnt") / 30.0 + 1e-6)).alias("ob_recent_prev_rate"),
        ((pl.col("ob_recent15_vol") / 15.0) / (pl.col("ob_prev30_vol") / 30.0 + 1e-6)).alias("ob_recent_prev_vol"),
        (pl.col("ob_recent15_sv") / (pl.col("ob_recent15_vol") + 1.0)).alias("ob_recent15_sv_imb"),
        ((pl.col("tb_buy_cnt") - pl.col("tb_sell_cnt")) / (pl.col("tb_trade_cnt") + 1.0)).alias("tb_cnt_imb"),
        (pl.col("tb_signed_vol") / (pl.col("tb_buy_vol") + pl.col("tb_sell_vol") + 1.0)).alias("tb_sv_imb"),
        (pl.col("tb_iat_std") / (pl.col("tb_iat_mean") + 1e-6)).alias("tb_iat_cv"),
        (pl.col("tb_iat_max") / (pl.col("tb_iat_mean") + 1e-6)).alias("tb_burst_ratio"),
        ((pl.col("tb_recent15_cnt") / 15.0) / (pl.col("tb_prev30_cnt") / 30.0 + 1e-6)).alias("tb_recent_prev_rate"),
        ((pl.col("tb_recent15_vol") / 15.0) / (pl.col("tb_prev30_vol") / 30.0 + 1e-6)).alias("tb_recent_prev_vol"),
        (pl.col("tb_recent15_sv") / (pl.col("tb_recent15_vol") + 1.0)).alias("tb_recent15_sv_imb"),
        (pl.col("tb_px_max") - pl.col("tb_px_min")).alias("tb_px_range"),
    ])
    # 派生的 log/ratio 特征 (数值稳定性: 全部加 1 或 MAD 归一化在 nuisance 阶段做)
    lab2 = lab2.with_columns([
        pl.col("ob_iat_mean").log1p().alias("ob_iat_mean_log"),
        pl.col("tb_iat_mean").log1p().alias("tb_iat_mean_log"),
        pl.col("ob_event_cnt").log1p().alias("ob_event_cnt_log"),
        pl.col("tb_trade_cnt").log1p().alias("tb_trade_cnt_log"),
        (pl.col("ob_add_vol") / (pl.col("ob_cancel_vol") + 1.0)).alias("ob_add_cancel_vol_ratio"),
    ])

    drop = ["ob_add_sv", "ob_cancel_sv"]  # 与 imb 重复
    lab2 = lab2.drop(drop)
    lab2.write_parquet(OUT / "raw_ot_agg.parquet")
    n_feat = len([c for c in lab2.columns if c.startswith(("ob_", "tb_"))])
    print(f"saved {lab2.shape} with {n_feat} O/T aggregate features "
          f"({time.time()-t0:.0f}s) → {OUT / 'raw_ot_agg.parquet'}")
    print("nulls:", lab2.null_count().sum_horizontal().item())


if __name__ == "__main__":
    main()
