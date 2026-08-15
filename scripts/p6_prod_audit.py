# -*- coding: utf-8 -*-
"""生产推理审计: test 侧资产盘点 (raw/processed/production preds)."""
import os
from pathlib import Path

RAW = Path(r"D:\mscapital-forecasting\data\raw")
PROC = Path(r"D:\mscapital-forecasting\data\processed")
OUT = Path(r"D:\mscapital-kaggle\output")

print("=== raw/test ===")
t = RAW / "test"
if t.exists():
    for f in sorted(t.iterdir()):
        print(f"  {f.name}  {f.stat().st_size/1e6:.0f}MB")
else:
    print("  MISSING")

print("\n=== raw/train ===")
for f in sorted((RAW / "train").iterdir()):
    print(f"  {f.name}  {f.stat().st_size/1e6:.0f}MB")

print("\n=== processed ===")
if PROC.exists():
    for f in sorted(PROC.iterdir()):
        print(f"  {f.name}  {f.stat().st_size/1e6:.0f}MB")
else:
    print("  MISSING")

print("\n=== 生产 baseline 预测 (c4_protocol_closed_final) ===")
base = OUT / "c4_protocol_closed_final" / "clean-baseline-v2"
if base.exists():
    for sub in sorted(base.iterdir()):
        p = base / sub
        files = [f.name for f in p.iterdir()] if p.is_dir() else []
        print(f"  {sub}: {files}")

print("\n=== c4_frozen_final ===")
base2 = OUT / "c4_frozen_final" / "clean-baseline-v2"
if base2.exists():
    for sub in sorted(base2.iterdir()):
        p = base2 / sub
        files = [f.name for f in p.iterdir()] if p.is_dir() else []
        print(f"  {sub}: {files}")

print("\n=== submissions 最新 ===")
subs = sorted((OUT / "submissions").iterdir(), key=lambda p: p.stat().st_mtime)
for f in subs[-6:]:
    print(f"  {f.name}  {f.stat().st_mtime}")

print("\n=== rlps (realmlp 生产推理痕迹) ===")
for d in ("rlps_final", "rlps_v12", "rlps_k9"):
    p = OUT / d
    if p.exists():
        print(f"  {d}: {[f.name for f in p.iterdir()]}")
