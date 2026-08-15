# -*- coding: utf-8 -*-
"""P6-PROD step 1: test raw O/T aggregates (复用 p5b_build_features 的 chunk 函数).

test 无 label → sample_id 取自 f0726_test。输出 output/p6_prod/raw_ot_agg_test.parquet
"""
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, r"D:\mscapital-kaggle\scripts")
import p5b_build_features as B

RAW = Path(r"D:\mscapital-forecasting\data\raw")
OUT = Path(r"D:\mscapital-kaggle\output\p6_prod")
OUT.mkdir(parents=True, exist_ok=True)
CHUNK = 200_000

t0 = time.time()
ids = pl.read_parquet(r"D:\mscapital-forecasting\data\processed\f0726_test_f32.parquet",
                      columns=["sample_id"])["sample_id"].to_numpy()
print(f"test ids: {len(ids):,} ({time.time()-t0:.0f}s)", flush=True)
lo, hi = int(ids.min()), int(ids.max())

olf = pl.scan_ipc(RAW / "test" / "order.feather", memory_map=False)
tlf = pl.scan_ipc(RAW / "test" / "transaction.feather", memory_map=False)

o_parts, t_parts = [], []
for a in range(lo, hi + 1, CHUNK):
    b = min(a + CHUNK - 1, hi)
    o_parts.append(B.build_order_chunk(olf, a, b))
    t_parts.append(B.build_trade_chunk(tlf, a, b))
    print(f"chunk {a:,}-{b:,} done ({time.time()-t0:.0f}s)", flush=True)

o_all = pl.concat(o_parts).sort("sample_id")
t_all = pl.concat(t_parts).sort("sample_id")
lab2 = pl.DataFrame({"sample_id": ids}).join(o_all, on="sample_id", how="left") \
    .join(t_all, on="sample_id", how="left")
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
lab2 = lab2.with_columns([
    pl.col("ob_iat_mean").log1p().alias("ob_iat_mean_log"),
    pl.col("tb_iat_mean").log1p().alias("tb_iat_mean_log"),
    pl.col("ob_event_cnt").log1p().alias("ob_event_cnt_log"),
    pl.col("tb_trade_cnt").log1p().alias("tb_trade_cnt_log"),
    (pl.col("ob_add_vol") / (pl.col("ob_cancel_vol") + 1.0)).alias("ob_add_cancel_vol_ratio"),
]).drop(["ob_add_sv", "ob_cancel_sv"])
lab2 = lab2.with_columns(pl.all().fill_null(0.0))
lab2.write_parquet(OUT / "raw_ot_agg_test.parquet")
n_feat = len([c for c in lab2.columns if c.startswith(("ob_", "tb_"))])
print(f"saved {lab2.shape} with {n_feat} features ({time.time()-t0:.0f}s)")
