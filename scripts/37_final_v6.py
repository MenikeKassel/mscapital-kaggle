# -*- coding: utf-8 -*-
"""
Final v6: 表格(R2+micro 全量) × TCN(全量) 融合提交
权重: w_tcn=0.07 (3-fold 验证最优 0.06-0.08 居中)
"""
import numpy as np
import polars as pl

P12 = r"D:\mscapital-forecasting\data\processed\p12_out"
OUT = r"D:\mscapital-kaggle\output\submissions\submission_blend_v6.csv"
W_TCN = 0.07

d1 = np.load(f"{P12}/v5_table_test_pred.npz")
d2 = np.load(f"{P12}/p12_full_test_pred.npz")
p_tab = d1["pred"]
p_tcn = d2["pred"]
te_ids = d2["test_ids"]
print(f"tabular {p_tab.shape} tcn {p_tcn.shape}")
print(f"corr(tab, tcn) on test = {np.corrcoef(p_tab, p_tcn)[0,1]:.4f}")

p_final = (1 - W_TCN) * p_tab + W_TCN * p_tcn
sub = pl.DataFrame({"sample_id": pl.Series(te_ids, dtype=pl.Int32),
                    "prediction": pl.Series(p_final, dtype=pl.Float64)}).sort("sample_id")
sub.write_csv(OUT)
print(f"\nsaved {OUT} ({sub.height:,} rows) NaN={sub['prediction'].is_null().sum()}")
print(f"pred: mean={sub['prediction'].mean():.6f} std={sub['prediction'].std():.6f}")
