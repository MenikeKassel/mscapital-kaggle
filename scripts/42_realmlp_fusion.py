# -*- coding: utf-8 -*-
"""
P1-1g: RealMLP × 表格融合验证 + v7 提交生成
关键: test 上 corr(realmlp, v5表格) — 吸取 v6 教训 (corr<0.1 弃用)
"""
import numpy as np
import polars as pl

P12 = r"D:\mscapital-forecasting\data\processed\p12_out"
OUT = r"D:\mscapital-kaggle\output\submissions"

d1 = np.load(f"{P12}/v5_table_test_pred.npz")
d2 = np.load(f"{P12}/realmlp_test_pred.npz")
p_tab = d1["pred"]
p_rl = d2["pred"]
te_ids = d2["test_ids"]
print(f"v5 tabular: {p_tab.shape} | realmlp: {p_rl.shape}")

c = np.corrcoef(p_tab, p_rl)[0, 1]
print(f"corr(v5表格, RealMLP) on test = {c:.4f}")

# 融合网格
best = (-1, None)
print("\n=== 融合搜索 ===")
for wr in np.arange(0, 0.41, 0.05):
    # 无法在 test 上算 cos (无 target), 用统计检查 + 网格存档
    p = (1 - wr) * p_tab + wr * p_rl
    # 打印统计 (尺度合理性)
    print(f"  w_rl={wr:.2f}: mean={p.mean():.6f} std={p.std():.6f}")

# 参考校准: RealMLP 单模型 LB 0.134 (公开), 我们的 v5 0.125
# 融合目标: 如果 corr ~0.4-0.7 → 融合增益预期 +0.003~0.006
print(f"\n判定: corr={c:.4f}")
if c < 0.1:
    print("!!! corr 过低 → 弃用 (v6 教训)")
elif c < 0.85:
    print("✅ 融合可行 (低相关互补)")

# 用 PSEUDO fold 验证融合权重 (RealMLP 在 PSEUDO 的 pred 没有——用作者 CV 验证集近似?)
# 保守: w_rl=0.15~0.25 (RealMLP 单模型强于我们任一单模型)
for wr in [0.15, 0.2, 0.25]:
    p = (1 - wr) * p_tab + wr * p_rl
    sub = pl.DataFrame({"sample_id": pl.Series(te_ids, dtype=pl.Int32),
                        "prediction": pl.Series(p, dtype=pl.Float64)}).sort("sample_id")
    fp = f"{OUT}/submission_blend_v7_rl{int(wr*100)}.csv"
    sub.write_csv(fp)
    print(f"saved {fp} (w_rl={wr})")
