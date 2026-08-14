# -*- coding: utf-8 -*-
"""Sample order/transaction value distributions (P5-B feature design)."""
import polars as pl

RAW = r"D:\mscapital-forecasting\data\raw\train"

o = pl.scan_ipc(rf"{RAW}\order.feather")
print("order sample:")
print(o.head(8).collect())
print("side values:", o.select(pl.col("side").unique()).collect().to_series().to_list())
print("order_action values:", o.select(pl.col("order_action").unique()).collect().to_series().to_list())
print("price/volume stats:", o.select(
    pl.col("price").min().alias("pmin"), pl.col("price").max().alias("pmax"),
    pl.col("volume").min().alias("vmin"), pl.col("volume").max().alias("vmax")).collect().row(0))
print("seconds distribution:", o.select(
    pl.col("seconds_before_predict").quantile(0.05).alias("q05"),
    pl.col("seconds_before_predict").quantile(0.5).alias("q50"),
    pl.col("seconds_before_predict").quantile(0.95).alias("q95")).collect().row(0))
# per-sample event counts
cnt = o.group_by("sample_id").len().collect()
print("events/sample: min=%d median=%d max=%d" % (
    cnt["len"].min(), cnt["len"].median(), cnt["len"].max()))

t = pl.scan_ipc(rf"{RAW}\transaction.feather")
print("\ntransaction sample:")
print(t.head(8).collect())
print("side values:", t.select(pl.col("side").unique()).collect().to_series().to_list())
print("price/volume stats:", t.select(
    pl.col("price").min().alias("pmin"), pl.col("price").max().alias("pmax"),
    pl.col("volume").min().alias("vmin"), pl.col("volume").max().alias("vmax")).collect().row(0))
cnt2 = t.group_by("sample_id").len().collect()
print("trades/sample: min=%d median=%d max=%d" % (
    cnt2["len"].min(), cnt2["len"].median(), cnt2["len"].max()))
