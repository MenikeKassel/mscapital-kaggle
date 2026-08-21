# -*- coding: utf-8 -*-
"""追加 P6-R20260821 到 registry.csv (28 列, 与 P6-01 同构)."""
import csv
from pathlib import Path

row = [
    "P6-R20260821", "canonical",
    "Z 线生产协议重跑 (今日验证数字)", "p6-r-20260821",
    "P6_production", "2026-08-21", "completed", "YELLOW",
    "P6-01", "", "P6-R", "production",
    "", "", "", "realmlp(3x256) C-05 E0",
    "MSE", "PSEUDO (inner 0-20/tune 21-32/refit 0-32/eval 33-70) + frozen 51-70",
    "0.142550 (canonical PSEUDO)",
    "RealMLP-C 0.139248; blendΔ +0.001460 (w=0.70 tuned 21-32)",
    "+0.001460",
    "55656536 (submitted, 等出分)",
    "重跑 Z 线生产协议: RealMLP-C 152+73Z 同源校准 + test 门禁全部复现 (standalone 0.139248/canon 0.142550/std ratio 0.9726 与 8/15 逐位一致, 确定性); blendΔ 今日 +0.001460 略高于 8/15 +0.001435; 三窗口 w 敏感性峰值 51-60 w=0.55 / 61-70 w=0.65 / PSEUDO w=0.55 → 生产 blend w=0.55; test 侧 corr(RealMLP-C,v8b)=0.9280 结构一致",
    "", "", "scripts/p6_prod_realmlp.py scripts/p6_final_w055.py scripts/p6_w_sensitivity.py",
    "docs/p6-production-inference.md",
    "output/p6_prod/{rerun_20260821.log, submission_candidate_p6_w055.csv, blend_candidates.npz}",
]

path = Path(r"D:\mscapital-kaggle\experiments\registry.csv")
# 注意: 首行已含 BOM, 追加行必须用纯 utf-8 (utf-8-sig 会在行中插入第二个 BOM)
with open(path, "a", encoding="utf-8", newline="") as f:
    csv.writer(f).writerow(row)
print("appended. last line:")
print(open(path, encoding="utf-8-sig").readlines()[-1][:120])
