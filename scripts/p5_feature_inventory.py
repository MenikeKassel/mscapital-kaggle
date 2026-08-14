# -*- coding: utf-8 -*-
"""Feature inventory for P5-B: what O/T aggregate features already exist."""
import polars as pl

OUT = r"D:\mscapital-kaggle\output"

# f0726 (153 features) — full column list grouped by prefix
fe = pl.read_parquet(rf"{OUT}\..\..\mscapital-forecasting\data\processed\f0726_train_f32.parquet",
                     columns=None) if False else None
import pyarrow.parquet as pq
pf = pq.read_schema(r"D:\mscapital-forecasting\data\processed\f0726_train_f32.parquet")
cols = [c for c in pf.names if c != "sample_id"]
print(f"f0726 features: {len(cols)}")
from collections import Counter
pref = Counter(c.split("_")[0] for c in cols)
print("  prefix counts:", dict(pref.most_common(20)))
print("  all:", cols)

for name, path in [
    ("m01a_features", rf"{OUT}\m01a_features"),
    ("p4_02c_features", rf"{OUT}\p4_02c_features"),
    ("m04_features", rf"{OUT}\m04_features"),
    ("p4_06a_features", rf"{OUT}\p4_06a_features"),
    ("p4_07_features", rf"{OUT}\p4_07_features"),
]:
    import os
    d = path
    if not os.path.isdir(d):
        print(f"\n{name}: DIR MISSING")
        continue
    files = os.listdir(d)
    print(f"\n{name}: {files}")
    for f in files[:3]:
        p = os.path.join(d, f)
        if f.endswith(".parquet"):
            try:
                s = pq.read_schema(p)
                print(f"  {f}: {s.names[:20]}... ({len(s.names)} cols)")
            except Exception as e:
                print(f"  {f}: err {e}")
        elif f.endswith(".npz"):
            import numpy as np
            z = np.load(p, allow_pickle=True)
            print(f"  {f}: keys={list(z.keys())} shapes={ {k: z[k].shape for k in z.keys()} }")

# market_state columns
s = pq.read_schema(rf"{OUT}\m05_state\market_state_train.parquet")
print("\nmarket_state_train cols:", s.names)
