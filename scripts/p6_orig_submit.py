# -*- coding: utf-8 -*-
"""组装纯原创生产候选: v5表格(R2+22micro, 纯原创) + RealMLP-C(152+73Z).
不使用 lb142 任何成分. 权重由 PSEUDO eval 33-70 调权 + test 侧门禁确认.
"""
import numpy as np
import polars as pl
from pathlib import Path

OUT = Path(r"D:\mscapital-kaggle\output\p6_prod")

def rms_norm(p):
    s = float(np.sqrt(np.mean(p ** 2)))
    return p / s if s > 0 else p

# ---- test 侧输入 (已确认 sample_id 同序) ----
p_v5 = np.load(r"D:\mscapital-forecasting\data\processed\p12_out\v5_table_test_pred.npz")["pred"]
c = np.load(OUT / "realmlpC_test_pred.npz")
sample_id = c["sample_id"]
p_rlc = c["pred"]
assert len(p_v5) == len(p_rlc) == 647896

# ---- PSEUDO 侧调权 (eval 33-70, v5表格 + RealMLP-C) ----
tbl = np.load(r"D:\mscapital-kaggle\output\rlps_v12\v5_table_pseudo_pred.npz")
rlc = np.load(OUT / "realmlpC_pseudo_pred.npz")
pt, y = tbl["pred"], tbl["y"]
pc = rlc["pred"]
assert len(pt) == len(pc) == len(y)

def cosine(p, y):
    return float(np.dot(p, y) / np.sqrt(np.dot(p, p) * np.dot(y, y)))

y = y.astype(np.float64)
cos_tbl = cosine(pt, y)
print(f"PSEUDO: v5表格={cos_tbl:.6f}")
best_w, best_d = None, -1e9
print("权重扫描 (v5 + w*RealMLP-C):")
for w in np.arange(0.05, 1.01, 0.05):
    pb = rms_norm(pt) + w * rms_norm(pc)
    d = cosine(pb, y) - cos_tbl
    if d > best_d:
        best_d, best_w = d, float(w)
print(f"  best w={best_w:.2f} blendΔ={best_d:+.6f}")

# ---- test 侧门禁 ----
# 用 61-70 valid std 对照
sp61 = np.load(r"D:\mscapital-kaggle\output\p5e_realmlp_spotcheck\C\R61_70\predictions.npz")
std_valid = float(np.std(sp61["pred"]))
# 主候选 w: PSEUDO best; 另存 w=0.20 作为保守对照(v7式)
print("\ntest 侧门禁:")
for w, tag in [(0.20, "conserv_v7式"), (best_w, "pseudo_best")]:
    p_main = rms_norm(p_v5) + w * rms_norm(p_rlc)
    std_t = float(np.std(p_main))
    ratio = std_t / std_valid
    corr = float(np.corrcoef(p_main, p_rlc)[0, 1])
    # 与 v8b 的 corr (结构对照)
    v8b = pl.read_csv(r"D:\mscapital-kaggle\output\submissions\submission_v8_ref50.csv")
    p_v8b = v8b["prediction"].to_numpy()
    corr_v8 = float(np.corrcoef(p_main, p_v8b)[0, 1])
    ok = "OK" if (0.8 < ratio < 1.3 and np.isfinite(p_main).all()) else "CHECK"
    print(f"  w={w:.2f} ({tag}): std_test={std_t:.5f} ratio={ratio:.4f} corr(_,RLC)={corr:.4f} corr(_,v8b)={corr_v8:.4f} [{ok}]")
    if tag == "pseudo_best":
        fn = OUT / "submission_orig_p6.csv"
    else:
        fn = OUT / f"submission_orig_p6_w{w:.2f}.csv"
    pl.DataFrame({"sample_id": sample_id, "prediction": p_main}).write_csv(fn)
    print(f"  wrote {fn} ({sum(1 for _ in open(fn, encoding='utf-8')):,} rows)")
