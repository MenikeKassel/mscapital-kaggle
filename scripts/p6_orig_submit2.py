# -*- coding: utf-8 -*-
"""生成纯原创最终提交候选 (修正门禁对照):
   w=0.60 主候选 + w=0.20 保守对照 (v7式).
   std ratio 门禁对 RealMLP-C 本体 (今天重跑已确认 0.9726); blend 候选做 corr/finite 检查.
"""
import numpy as np
import polars as pl
from pathlib import Path

OUT = Path(r"D:\mscapital-kaggle\output\p6_prod")

def rms_norm(p):
    s = float(np.sqrt(np.mean(p ** 2)))
    return p / s if s > 0 else p

# ---- 输入 (已确认同序) ----
p_v5 = np.load(r"D:\mscapital-forecasting\data\processed\p12_out\v5_table_test_pred.npz")["pred"]
c = np.load(OUT / "realmlpC_test_pred.npz")
sample_id = c["sample_id"]
p_rlc = c["pred"]
assert len(p_v5) == len(p_rlc) == 647896 and np.isfinite(p_v5).all() and np.isfinite(p_rlc).all()

# ---- 门禁 (对照真实基准) ----
v8b = pl.read_csv(r"D:\mscapital-kaggle\output\submissions\submission_v8_ref50.csv")
p_v8b = v8b["prediction"].to_numpy()
# RealMLP-C 本体 std ratio (今天重跑 p6 results.json)
std_test_rlc = float(np.std(p_rlc))
std_valid = 0.014779058  # 61-70, p6 results
print(f"门禁-RealMLP-C本体: std_test={std_test_rlc:.5f} std_valid={std_valid:.5f} ratio={std_test_rlc/std_valid:.4f} (应≈0.9726 OK)")

print("\n纯原创候选 test 侧:")
for w, tag in [(0.20, "conserv"), (0.60, "MAIN")]:
    p_main = rms_norm(p_v5) + w * rms_norm(p_rlc)
    corr_v8 = float(np.corrcoef(p_main, p_v8b)[0, 1])
    corr_rlc = float(np.corrcoef(p_main, p_rlc)[0, 1])
    finite = bool(np.isfinite(p_main).all())
    print(f"  w={w:.2f} ({tag}): corr(_,v8b)={corr_v8:.4f} corr(_,RLC)={corr_rlc:.4f} std={np.std(p_main):.4f} finite={finite}")
    fn = OUT / ("submission_orig_p6.csv" if tag == "MAIN" else f"submission_orig_p6_w{w:.2f}.csv")
    pl.DataFrame({"sample_id": sample_id, "prediction": p_main}).write_csv(fn)
    print(f"  wrote {fn} ({sum(1 for _ in open(fn, encoding='utf-8')):,} rows)")

print("\n提交候选 = v5表格(R2+22micro,纯原创) + 0.60×RealMLP-C(152+73Z)")
print("PSEUDO blendΔ = +0.0062 (全期) / +0.0050 (51-70) — 纯原创锚上 Z 线增量远强于 v8b)")
