# -*- coding: utf-8 -*-
"""验证原始数据: 行数/列名/类型, 对照官方schema"""
import polars as pl
import time

RAW = r"D:\mscapital-forecasting\data\raw"

files = [
    ("train/label.feather", None),
    ("train/market.feather", None),
    ("train/order.feather", None),
    ("train/transaction.feather", None),
    ("test/market.feather", None),
    ("test/order.feather", None),
    ("test/transaction.feather", None),
]

for path, _ in files:
    t0 = time.time()
    lf = pl.scan_ipc(f"{RAW}/{path}", memory_map=False)
    cols = lf.collect_schema().names()
    dtypes = [str(d) for d in lf.collect_schema().dtypes()]
    n = lf.select(pl.len()).collect().item()
    print(f"{path}: rows={n:,} cols={len(cols)} ({time.time()-t0:.1f}s)")
    print(f"   {list(zip(cols, dtypes))}")

# label 月份分布 + target 基本统计
lab = pl.read_ipc(f"{RAW}/train/label.feather", memory_map=False)
print("\nlabel: month range", lab["month"].min(), "-", lab["month"].max(),
      "| n_unique months:", lab["month"].n_unique())
print("target: mean=%.6f std=%.6f min=%.4f max=%.4f" % (
    lab["target"].mean(), lab["target"].std(), lab["target"].min(), lab["target"].max()))
print("target>0 ratio:", (lab["target"] > 0).mean())
print("sample_id unique:", lab["sample_id"].n_unique(), "/", lab.height)

sub = pl.read_csv(f"{RAW}/submission.csv")
print("\nsubmission.csv:", sub.shape, sub.columns)
