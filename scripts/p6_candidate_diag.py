# -*- coding: utf-8 -*-
"""候选在测试集侧的统计 (无 label 可得的全部证据)."""
import numpy as np
import polars as pl

d = np.load(r"D:\mscapital-kaggle\output\p6_prod\blend_candidates.npz")
p_v8b, p_c, p_main = d["p_v8b"], d["p_realmlpC"], d["p_main"]

print("=== 测试集侧诊断 (647,896 行, label 隐藏) ===")
print(f"v8b:      std={p_v8b.std():.5f}  mean={p_v8b.mean():+.6f}")
print(f"RealMLP-C: std={p_c.std():.5f}   mean={p_c.mean():+.6f}")
print(f"候选(w=0.55): std={p_main.std():.5f} mean={p_main.mean():+.6f}")
print(f"corr(RealMLP-C, v8b) on test = {np.corrcoef(p_c, p_v8b)[0,1]:.4f}")
print(f"corr(候选, v8b) on test     = {np.corrcoef(p_main, p_v8b)[0,1]:.4f}")

# 与本地验证窗口的 corr 结构对比: 61-70 上 corr(RealMLP-C, canonical)=0.940
# 若 test 上 corr 结构与 61-70 类似 → blend 增益可外推
for w in (0.3, 0.4, 0.5, 0.6, 0.7):
    p_w = p_v8b / np.sqrt(np.mean(p_v8b**2)) + w * p_c / np.sqrt(np.mean(p_c**2))
    print(f"w={w:.1f}: std={p_w.std():.5f} corr(cand,v8b)={np.corrcoef(p_w, p_v8b)[0,1]:.4f}")

# 候选 vs v8b 的差异幅度 (改动有多大)
diff = p_main - p_v8b / np.sqrt(np.mean(p_v8b**2))
print(f"\n候选相对 v8b(rms) 的改动: mean|Δ|={np.abs(diff).mean():.5f} (v8b std={p_v8b.std():.5f})")
print(f"改动占比: {np.abs(diff).mean()/p_v8b.std():.2%} 的 v8b std")
