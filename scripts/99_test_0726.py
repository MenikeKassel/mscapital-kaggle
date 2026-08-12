# -*- coding: utf-8 -*-
"""测试: 0726 特征构建 1 块 (4000 样本)"""
import sys, time
import polars as pl

sys.path.insert(0, r"D:\mscapital-forecasting\reference")
import rfmf_0726data_source as src

src.BASE_PATH = r"D:\mscapital-forecasting\data\raw"
src.PROCESSED_PATH = r"D:\mscapital-forecasting\data\processed\0726_secondly"
src.OUTPUT_PATH = src.PROCESSED_PATH

t0 = time.time()
src.main()
print(f"resample done ({time.time()-t0:.0f}s)", flush=True)
import os
print("secondly files:", os.listdir(src.PROCESSED_PATH), flush=True)

t0 = time.time()
p = src.get_data('train', start_id=0, end_id=4000, return_pandas=False)
print(f"get_data 4000 samples: {p.shape} ({time.time()-t0:.0f}s)", flush=True)
print(f"cols: {len(p.columns)}")
print(p.head(2).select([c for c in p.columns if c != 'sample_id'][:5]).to_dict(as_series=False) if False else "ok")
print("TEST PASS", flush=True)
