# -*- coding: utf-8 -*-
import json
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from pathlib import Path

OUT = Path(r"D:\mscapital-kaggle\output")

print("=== m05_state 目录 ===")
p = OUT / "m05_state"
for f in sorted(p.iterdir()):
    print(f"  {f.name}  {f.stat().st_size/1e6:.0f}MB")

print("\n=== PSEUDO manifest (split) ===")
m = json.loads((OUT / "c4_protocol_closed_final/clean-baseline-v2/PSEUDO/manifest.json").read_text(encoding="utf-8"))
print(json.dumps(m, indent=1)[:1500])

print("\n=== production canonical_scale_predictions ===")
d = np.load(OUT / "c4_protocol_closed_final/clean-baseline-v2/production/canonical_scale_predictions.npz")
print("  keys:", list(d.keys()))
for k in d.keys():
    print(f"  {k}: shape={d[k].shape}")

print("\n=== production_scales.json ===")
print(json.loads((OUT / "c4_protocol_closed_final/clean-baseline-v2/production/production_scales.json").read_text(encoding="utf-8")))

print("\n=== f0726_test_f32 列 ===")
s = pq.read_schema(r"D:\mscapital-forecasting\data\processed\f0726_test_f32.parquet")
print(f"  {len(s.names)} cols, first 5: {s.names[:5]}, last 3: {s.names[-3:]}")
print("  含 target?", "target" in s.names, "| 含 sample_id?", "sample_id" in s.names)

print("\n=== test label/sample 数 (submission 行数参照) ===")
sub = pl.read_csv(OUT / "submissions/submission_v9_cos_a17.csv")
print(f"  submission: {sub.shape} cols={sub.columns}")

print("\n=== f0726_test 行数 ===")
tf = pl.read_parquet(r"D:\mscapital-forecasting\data\processed\f0726_test_f32.parquet", columns=["sample_id"])
print(f"  rows={tf.height} sample_id range={tf['sample_id'].min()}-{tf['sample_id'].max()}")
