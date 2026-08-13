# -*- coding: utf-8 -*-
"""
Clean Baseline v2 校准提交: 生产融合
realmlp_prod_test_pred.npz (云端) + clean_table_test_pred.npz (本地)
→ apply_production_rule (frozen: rms, table 0.37) → submission
"""
import sys
import numpy as np
import polars as pl

sys.path.insert(0, r"D:\mscapital-kaggle\src")
from mscapital.clean_baseline import apply_production_rule

P12 = r"D:\mscapital-forecasting\data\processed\p12_out"
OUT = r"D:\mscapital-kaggle\output\submissions"

# frozen scales (production_scales.json)
SCALE_REALMLP = 0.014812425837302948
SCALE_TABLE = 0.00035057231611417754
TABLE_WEIGHT = 0.37

d1 = np.load(f"{P12}/realmlp_prod_test_pred.npz")
d2 = np.load(f"{P12}/clean_table_test_pred.npz")
p_rl, ids_rl = d1["pred"], d1["test_ids"]
p_tab, ids_tab = d2["pred"], d2["test_ids"]
print(f"realmlp: {p_rl.shape} (mean={p_rl.mean():.2e} std={p_rl.std():.2e})")
print(f"table:   {p_tab.shape} (mean={p_tab.mean():.2e} std={p_tab.std():.2e})")
print(f"corr(realmlp, table) on test: {np.corrcoef(p_rl, p_tab)[0,1]:.4f}")

# 对齐 sample_id
assert np.array_equal(ids_rl, ids_tab), "test_ids mismatch"
p_final = apply_production_rule(p_rl, p_tab, scale_realmlp=SCALE_REALMLP,
                                scale_table=SCALE_TABLE, table_weight=TABLE_WEIGHT)
print(f"final: mean={p_final.mean():.2e} std={p_final.std():.2e}")

sub = pl.DataFrame({"sample_id": pl.Series(ids_rl, dtype=pl.Int32),
                    "prediction": pl.Series(p_final, dtype=pl.Float64)}).sort("sample_id")
fp = f"{OUT}/submission_cbv2_calib.csv"
sub.write_csv(fp)
print(f"saved {fp} ({len(sub):,} rows)")
