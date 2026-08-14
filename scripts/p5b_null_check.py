# -*- coding: utf-8 -*-
import polars as pl

OUT = r"D:\mscapital-kaggle\output"
PROC = r"D:\mscapital-forecasting\data\processed"

for name, p in [
    ("f0726", rf"{PROC}\f0726_train_f32.parquet"),
    ("rawagg", rf"{OUT}\p5b_scfi\raw_ot_agg.parquet"),
    ("mstate", rf"{OUT}\m05_state\market_state_train.parquet"),
]:
    df = pl.read_parquet(p)
    nc = df.null_count()
    total = int(nc.sum_horizontal().item())
    if total:
        bad = [(c, int(nc[c].item())) for c in df.columns if int(nc[c].item()) > 0]
        print(f"{name}: {total} nulls in {len(bad)} cols, top:", bad[:8])
    else:
        print(f"{name}: no nulls")
