# -*- coding: utf-8 -*-
"""压缩 0726 特征为 float32 (轻量操作, 2分钟)"""
import polars as pl

for mode, path in [("train", r"D:\mscapital-forecasting\data\processed\f0726_train.parquet"),
                   ("test", r"D:\mscapital-forecasting\data\processed\f0726_test.parquet")]:
    df = pl.read_parquet(path)
    feats = [c for c in df.columns if c not in ("sample_id", "target")]
    df2 = df.with_columns([pl.col(c).cast(pl.Float32) for c in feats])
    out = path.replace(".parquet", "_f32.parquet")
    df2.write_parquet(out, compression="zstd")
    import os
    print(f"{mode}: {df2.shape} -> {os.path.getsize(out)/1e6:.0f}MB", flush=True)
print("done")
