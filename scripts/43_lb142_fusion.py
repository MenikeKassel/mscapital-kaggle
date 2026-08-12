# -*- coding: utf-8 -*-
"""
P1-1i: LB142 预测 × 我们的 v7 融合分析
corr + 融合网格 → v8 提交文件
"""
import numpy as np
import polars as pl

P12 = r"D:\mscapital-forecasting\data\processed\p12_out"
REF = r"D:\mscapital-forecasting\reference\lb142\submission_ref_lb0142.csv"

ref = pl.read_csv(REF).sort("sample_id")
print(f"lb142 ref: {ref.shape}, cols: {ref.columns}")
print(ref.head(3))

p_ref = ref["prediction"].to_numpy().astype(np.float64)

# 我们的 v7 pred (= 0.8*v5 + 0.2*RealMLP)
d1 = np.load(f"{P12}/v5_table_test_pred.npz")
d2 = np.load(f"{P12}/realmlp_test_pred.npz")
p_v5 = d1["pred"]
p_rl = d2["pred"]
p_v7 = 0.8 * p_v5 + 0.2 * p_rl
te_ids = d2["test_ids"]

print(f"\ncorr(lb142_ref, v5表格) = {np.corrcoef(p_ref, p_v5)[0,1]:.4f}")
print(f"corr(lb142_ref, RealMLP) = {np.corrcoef(p_ref, p_rl)[0,1]:.4f}")
print(f"corr(lb142_ref, v7) = {np.corrcoef(p_ref, p_v7)[0,1]:.4f}")
print(f"\nscale: ref mean={p_ref.mean():.6f} std={p_ref.std():.6f}")
print(f"       v7  mean={p_v7.mean():.6f} std={p_v7.std():.6f}")

# 融合网格 (无 target, 用统计检查 + 多版本存档)
print("\n=== 融合版本存档 ===")
for wr in [0.3, 0.4, 0.5, 0.6, 0.7]:
    p = (1 - wr) * p_v7 + wr * p_ref
    sub = pl.DataFrame({"sample_id": pl.Series(te_ids, dtype=pl.Int32),
                        "prediction": pl.Series(p, dtype=pl.Float64)}).sort("sample_id")
    fp = rf"D:\mscapital-kaggle\output\submissions\submission_v8_ref{int(wr*100)}.csv"
    sub.write_csv(fp)
    print(f"  w_ref={wr}: saved v8_ref{int(wr*100)} (mean={p.mean():.6f} std={p.std():.6f})")
