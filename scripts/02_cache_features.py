# -*- coding: utf-8 -*-
"""
02: 特征缓存 — 构建一次 90特征, 存 parquet 供所有实验复用。
输出: D:/mscapital-forecasting/data/processed/{train,test}_features.parquet
"""
import gc, time, sys, os
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b1_official_baseline import market_feats, tx_feats, ord_feats, cross_feats

OUT_DIR = r"D:\mscapital-forecasting\data\processed"
RAW = r"D:\mscapital-forecasting\data\raw"
T0 = time.time()

def build(split):
    t0 = time.time()
    mf = market_feats(split); gc.collect()
    tf = tx_feats(split); gc.collect()
    of = ord_feats(split); gc.collect()
    df = mf.join(tf, on="sample_id", how="left").join(of, on="sample_id", how="left")
    df = cross_feats(df)
    print(f"{split}: {df.shape} ({time.time()-t0:.1f}s)", flush=True)
    return df

os.makedirs(OUT_DIR, exist_ok=True)
tr = build("train")
label = pl.read_ipc(f"{RAW}/train/label.feather", memory_map=False)
tr = tr.join(label, on="sample_id", how="inner")
tr.write_parquet(f"{OUT_DIR}/train_features.parquet")
print(f"train saved: {tr.shape}", flush=True)
del tr; gc.collect()

te = build("test")
te.write_parquet(f"{OUT_DIR}/test_features.parquet")
print(f"test saved: {te.shape}", flush=True)
print(f"TOTAL {time.time()-T0:.1f}s", flush=True)
