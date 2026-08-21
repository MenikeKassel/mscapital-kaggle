# -*- coding: utf-8 -*-
"""生成今日 P6 最终提交候选: v8b + 0.55*RealMLP-C (w=0.55 三窗口稳健).
读今天重跑产出的 blend_candidates.npz (含今天 fresh 的 p_realmlpC),
输出 submission_candidate_p6_w055.csv + 格式校验.
"""
import numpy as np
import polars as pl
from pathlib import Path

OUT = Path(r"D:\mscapital-kaggle\output\p6_prod")
d = np.load(OUT / "blend_candidates.npz")
p_v8b = d["p_v8b"]
p_c = d["p_realmlpC"]
sample_id = d["sample_id"]

def rms_norm(p):
    s = float(np.sqrt(np.mean(p ** 2)))
    return p / s if s > 0 else p

W = 0.55
p_main = rms_norm(p_v8b) + W * rms_norm(p_c)

# 分布校验
print(f"rows: {len(sample_id):,}  (expect 647,896)")
print(f"p_main std={p_main.std():.5f} mean={p_main.mean():+.6f}")
print(f"corr(p_main, v8b)={np.corrcoef(p_main, p_v8b)[0,1]:.4f}")
print(f"finite: {np.isfinite(p_main).all()}")

df = pl.DataFrame({"sample_id": sample_id, "prediction": p_main})
out = OUT / "submission_candidate_p6_w055.csv"
df.write_csv(out)
print(f"\nWRITTEN: {out}")
print(f"rows+header: {sum(1 for _ in open(out, encoding='utf-8')):,}")
