# -*- coding: utf-8 -*-
"""P3-03/P3-04 shared: per-second event grid from processed secondly data.

Grid: (60 seconds x 6 channels):
  ch0 bid_add_vol, ch1 ask_add_vol, ch2 bid_cancel_vol, ch3 ask_cancel_vol,
  ch4 trade_buy_vol, ch5 trade_sell_vol   (per second, aggregated)
Plus price-structure histogram: 16 log-price bins x 3 channels
  (order net flow, cancel volume, trade volume) over the full 60s window.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl

ORDER = Path(r"D:\kaggle\working\processed_data\train_order_secondly.feather")
TX = Path(r"D:\kaggle\working\processed_data\train_transaction_secondly.feather")
GRID_N_SEC = 60
N_CHANNELS = 6
PRICE_BINS = 16


def build_second_grid(order_path=ORDER, tx_path=TX) -> np.ndarray:
    """Return (n_samples, 60, 6) float32 grid, rows sorted by sample_id."""
    order = pl.read_ipc(order_path)
    tx = pl.read_ipc(tx_path)
    # order channels
    order = order.with_columns(
        pl.when((pl.col("side") == 0) & (pl.col("order_action") == 0)).then(pl.col("volume"))
        .otherwise(0.0).alias("bid_add"),
        pl.when((pl.col("side") == 1) & (pl.col("order_action") == 0)).then(pl.col("volume"))
        .otherwise(0.0).alias("ask_add"),
        pl.when((pl.col("side") == 0) & (pl.col("order_action") == 1)).then(pl.col("volume"))
        .otherwise(0.0).alias("bid_cancel"),
        pl.when((pl.col("side") == 1) & (pl.col("order_action") == 1)).then(pl.col("volume"))
        .otherwise(0.0).alias("ask_cancel"),
    ).group_by("sample_id", "seconds_before_predict").agg(
        pl.col("bid_add").sum(), pl.col("ask_add").sum(),
        pl.col("bid_cancel").sum(), pl.col("ask_cancel").sum(),
    )
    tx = tx.with_columns(
        pl.when(pl.col("side") == 0).then(pl.col("volume")).otherwise(0.0).alias("buy"),
        pl.when(pl.col("side") == 1).then(pl.col("volume")).otherwise(0.0).alias("sell"),
    ).group_by("sample_id", "seconds_before_predict").agg(
        pl.col("buy").sum(), pl.col("sell").sum(),
    )
    merged = order.join(tx, on=["sample_id", "seconds_before_predict"], how="outer").fill_null(0.0)
    merged = merged.sort(["sample_id", "seconds_before_predict"])
    ids = merged["sample_id"].to_numpy().astype(np.int64)
    sec = merged["seconds_before_predict"].to_numpy().astype(np.int64)
    ch = np.column_stack([
        merged["bid_add"].to_numpy(), merged["ask_add"].to_numpy(),
        merged["bid_cancel"].to_numpy(), merged["ask_cancel"].to_numpy(),
        merged["buy"].to_numpy(), merged["sell"].to_numpy(),
    ]).astype(np.float32)
    n = int(np.max(ids)) + 1
    grid = np.zeros((n, GRID_N_SEC, N_CHANNELS), dtype=np.float32)
    valid = (sec >= 0) & (sec < GRID_N_SEC)
    grid[ids[valid], sec[valid], :] = ch[valid]
    # drop sample ids with zero total rows? keep all; canonical alignment filters later
    return grid


def build_price_hist(order_path=ORDER, tx_path=TX) -> tuple[np.ndarray, np.ndarray]:
    """Return (n_samples, PRICE_BINS*3) histogram + global log-price bin edges."""
    order = pl.read_ipc(order_path)
    tx = pl.read_ipc(tx_path)
    all_prices = np.concatenate([
        np.log(order["price"].to_numpy() + 1e-8),
        np.log(tx["price"].to_numpy() + 1e-8),
    ])
    all_prices = all_prices[np.isfinite(all_prices)]
    edges = np.quantile(all_prices, np.linspace(0, 1, PRICE_BINS + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf

    n = max(int(order["sample_id"].max()), int(tx["sample_id"].max())) + 1
    h = np.zeros((n, PRICE_BINS * 3), dtype=np.float32)
    o_px = np.log(order["price"].to_numpy() + 1e-8)
    o_ids = order["sample_id"].to_numpy()
    o_valid = np.isfinite(o_px)
    o_bin = np.clip(np.digitize(o_px[o_valid], edges) - 1, 0, PRICE_BINS - 1)
    o_sign = np.where(order["order_action"].to_numpy()[o_valid] == 0, 1.0, -1.0) * np.where(order["side"].to_numpy()[o_valid] == 0, 1.0, -1.0)
    o_vol = order["volume"].to_numpy().astype(np.float32)[o_valid] * o_sign
    np.add.at(h[:, 0:PRICE_BINS], (o_ids[o_valid], o_bin), o_vol)
    t_px = np.log(tx["price"].to_numpy() + 1e-8)
    t_ids = tx["sample_id"].to_numpy()
    t_valid = np.isfinite(t_px)
    t_bin = np.clip(np.digitize(t_px[t_valid], edges) - 1, 0, PRICE_BINS - 1)
    t_vol = tx["volume"].to_numpy().astype(np.float32)[t_valid]
    np.add.at(h[:, PRICE_BINS:2 * PRICE_BINS], (t_ids[t_valid], t_bin), t_vol)
    np.add.at(h[:, 2 * PRICE_BINS:3 * PRICE_BINS], (o_ids[o_valid], o_bin), np.abs(o_vol))
    return h, edges
