# -*- coding: utf-8 -*-
"""P5-B 补充诊断: 逐月 C-vs-A (同 learner 对比), nuisance R2=1.0 特征定位,
corr 矩阵, C 臂 top-20 feature importance."""
import json
import numpy as np

OUT = r"D:\mscapital-kaggle\output\p5b_scfi"
d = np.load(rf"{OUT}\p5b_preds.npz")
month, y, canon = d["month"], d["y"], d["canon"]

def cos(p, y):
    return float(np.dot(p, y) / np.sqrt(np.dot(p, p) * np.dot(y, y)))

print("=== 逐月 C-vs-A (同 learner) ===")
pos = 0
for mm in sorted(set(month.tolist())):
    m = month == mm
    cA = cos(d["p_A"][m], y[m])
    cC = cos(d["p_C"][m], y[m])
    cB = cos(d["p_B"][m], y[m])
    cK = cos(canon[m], y[m])
    dC = cC - cA
    pos += dC > 0
    print(f"  m{mm}: A={cA:.5f} C={cC:.5f} Δ(C-A)={dC:+.5f} B={cB:.5f} canon={cK:.5f}")
print(f"  months C>A: {pos}/20")

print("\n=== corr 矩阵 (51-70 拼接) ===")
names = ["p_A", "p_B", "p_C", "p_D", "p_E", "canon"]
arr = np.column_stack([d[n] for n in names])
cmat = np.corrcoef(arr.T)
for i, ni in enumerate(names):
    print("  " + "  ".join(f"{ni} {cmat[i, j]:+.3f}" for j, nj in enumerate(names)))

print("\n=== nuisance R² 异常特征 (max) ===")
r = json.load(open(rf"{OUT}\results.json", encoding="utf-8"))
pf = r["nuisance"]["B51_60"]["per_feature"]
top = sorted(pf.items(), key=lambda kv: -kv[1])[:6]
for k, v in top:
    print(f"  {k}: R2={v:.4f}")

print("\n=== arm C top-20 gain (B51_60) ===")
imp = r["importance_C"]["B51_60"]
print("  z_in_top20:", imp["z_in_top20"])
print("  z_rank_best:", imp["z_rank_best"])
for n, v in imp["top20"][:12]:
    print(f"    {n}: {v:.1f}")

print("\n=== blend Δ 明细 ===")
for arm in "ABCDE":
    b1 = r["arms"][arm]["B51_60"]["blend_delta"]
    b2 = r["arms"][arm]["B61_70"]["blend_delta"]
    print(f"  {arm}: B1 {b1:+.6f} B2 {b2:+.6f} w=({r['arms'][arm]['B51_60']['blend_w']}, {r['arms'][arm]['B61_70']['blend_w']})")

print("\n=== criteria ===")
print(json.dumps(r["consolidated"]["criteria"], indent=1))
print("decision:", r["consolidated"]["decision"])
