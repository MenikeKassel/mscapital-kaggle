# -*- coding: utf-8 -*-
"""
三路融合分析: v7(v5+RealMLP) × f0726树 × lb142ref
corr 矩阵 + 融合网格存档 (不提交, 等用户拍板)
"""
import numpy as np
import polars as pl

P12 = r"D:\mscapital-forecasting\data\processed\p12_out"
OUT = r"D:\mscapital-kaggle\output\submissions"
REF = r"D:\mscapital-forecasting\reference\lb142\submission_ref_lb0142.csv"

# v5 表格 + RealMLP
d1 = np.load(f"{P12}/v5_table_test_pred.npz")
d2 = np.load(f"{P12}/realmlp_test_pred.npz")
p_v5, p_rl, te_ids = d1["pred"], d2["pred"], d2["test_ids"]
p_v7 = 0.8 * p_v5 + 0.2 * p_rl

# f0726 树
d3 = np.load(r"D:\mscapital-kaggle\output\f0726_v6b\f0726_tree_test_pred.npz")
p_tr = d3["pred"]
# 与 te_ids 对齐 (f0726 的 test_ids)
tr_ids = d3["test_ids"]
order = np.argsort(tr_ids)
p_tr = p_tr[order]
print(f"align: {len(te_ids)} vs {len(p_tr)}")

# lb142 ref
ref = pl.read_csv(REF).sort("sample_id")
p_ref = ref["prediction"].to_numpy().astype(np.float64)

# corr 矩阵
names = ["v5", "RealMLP", "v7", "f0726树", "lb142ref"]
Ps = [p_v5, p_rl, p_v7, p_tr, p_ref]
print("\ncorr 矩阵:")
print("          " + "".join(f"{n:>10}" for n in names))
for i, n in enumerate(names):
    row = "".join(f"{np.corrcoef(Ps[i], Ps[j])[0,1]:10.4f}" for j in range(len(names)))
    print(f"{n:>10}{row}")

# 融合: v7 × f0726树 (独立信息源)
print("\n=== v7 × f0726树 ===")
for wt in [0.05, 0.1, 0.15, 0.2]:
    p = (1 - wt) * p_v7 + wt * p_tr
    c1 = np.corrcoef(p, p_ref)[0, 1]
    print(f"  w_tree={wt}: corr(vs ref)={c1:.4f}")

# 三路: v7 × tree × ref
print("\n=== 三路融合 v7/tree/ref ===")
for wt in [0.05, 0.1]:
    for wr in [0.3, 0.4, 0.5]:
        p = (1 - wt - wr) * p_v7 + wt * p_tr + wr * p_ref
        sub = pl.DataFrame({"sample_id": pl.Series(te_ids, dtype=pl.Int32),
                            "prediction": pl.Series(p, dtype=pl.Float64)}).sort("sample_id")
        fp = f"{OUT}/submission_v9_t{int(wt*100)}r{int(wr*100)}.csv"
        sub.write_csv(fp)
        print(f"  w_tree={wt} w_ref={wr}: saved v9_t{int(wt*100)}r{int(wr*100)} (mean={p.mean():.2e} std={p.std():.4f})")
