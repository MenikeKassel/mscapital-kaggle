# -*- coding: utf-8 -*-
"""P5-E 追加: corr/blend 判据 (任务书 §5.12 LIVE 条).

blend 权重在 inner_tune (51-60) 调 (RealMLP inner_predictions vs canonical 51-60),
eval 61-70 (RealMLP outer predictions vs canonical 61-70). 完全 temporal。
"""
import numpy as np

BASE = r"D:\mscapital-kaggle\output\p5e_realmlp_spotcheck"
CANON = r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz"

c = np.load(CANON)
cids, cmonth, cbase = c["sample_id"], c["month"].astype(int), c["baseline_oof"]


def cosine(p, y):
    return float(np.dot(p, y) / np.sqrt(np.dot(p, p) * np.dot(y, y)))


def rms_norm(p):
    s = float(np.sqrt(np.mean(p ** 2)))
    return p / s if s > 0 else p


for arm in ("A", "C"):
    inner = np.load(rf"{BASE}\{arm}\R61_70\inner_predictions.npz")
    outer = np.load(rf"{BASE}\{arm}\R61_70\predictions.npz")
    # align canonical by sample_id
    i_pos = np.searchsorted(cids, inner["sample_id"])
    c_inner = cbase[i_pos]
    o_pos = np.searchsorted(cids, outer["sample_id"])
    c_outer = cbase[o_pos]
    y_i, y_o = inner["target"], outer["target"]
    p_i, p_o = inner["pred"], outer["pred"]
    # blend weight tuned on 51-60 (inner tune)
    best_w, best_s = 0.0, -1e9
    for w in np.arange(0.05, 0.51, 0.05):
        s = cosine(rms_norm(c_inner) + w * rms_norm(p_i), y_i)
        if s > best_s:
            best_s, best_w = s, float(w)
    b0 = cosine(c_outer, y_o)
    b1 = cosine(rms_norm(c_outer) + best_w * rms_norm(p_o), y_o)
    corr = float(np.corrcoef(p_o, c_outer)[0, 1])
    print(f"arm {arm}: outer cos={cosine(p_o, y_o):.6f} corr(canon)={corr:.4f} "
          f"blend_w={best_w:.2f} blendΔ={b1 - b0:+.6f} (canon cos={b0:.6f})")

# C vs A 正交性
ia = np.load(rf"{BASE}\A\R61_70\predictions.npz")["pred"]
ic = np.load(rf"{BASE}\C\R61_70\predictions.npz")["pred"]
print(f"corr(C,A preds) = {np.corrcoef(ia, ic)[0,1]:.4f}")
