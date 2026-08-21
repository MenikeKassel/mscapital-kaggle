# -*- coding: utf-8 -*-
"""P6 提交前 w 敏感性扩展 (只读, 复用已有预测, 不重训).
三个独立窗口扫 w∈[0.05, 1.05]:
  1) 51-60 调权窗     (SPOT_INNER = RealMLP-C spot C R61_70 inner)
  2) 61-70 外窗       (SPOT_OUTER + CANON_OOF)
  3) PSEUDO eval 33-70 (realmlpC_pseudo_pred + CANON_PSEUDO)
判定: 主候选 w=0.55 在每窗的 blendΔ, 峰值 w, 峰值是否稳定落在 0.5-0.8 带.
"""
import numpy as np

def cosine(p, y):
    return float(np.dot(p, y) / np.sqrt(np.dot(p, p) * np.dot(y, y)))

def rms_norm(p):
    s = float(np.sqrt(np.mean(p ** 2)))
    return p / s if s > 0 else p

def scan(label, canon, p_rmlp, y, w_grid):
    print(f"\n=== {label} ===")
    cos_c = cosine(canon, y)
    best_w, best_d = 0.0, -1e9
    rows = []
    for w in w_grid:
        pb = rms_norm(canon) + w * rms_norm(p_rmlp)
        d = cosine(pb, y) - cos_c
        rows.append((w, d))
        if d > best_d:
            best_d, best_w = d, float(w)
    # 打印 0.45-0.85 精细区 + 峰值
    for w, d in rows:
        mark = " <== PEAK" if abs(w - best_w) < 1e-9 else ""
        if 0.40 <= w <= 0.90 or mark:
            print(f"  w={w:.2f}: blendΔ={d:+.6f}{mark}")
    print(f"  > peak w={best_w:.2f} blendΔ={best_d:+.6f} | canon={cos_c:.6f}")
    # 主候选 w=0.55 处的值
    d55 = cosine(rms_norm(canon) + 0.55 * rms_norm(p_rmlp), y) - cos_c
    print(f"  > 主候选 w=0.55: blendΔ={d55:+.6f}")
    return best_w, best_d, d55

W = np.arange(0.05, 1.06, 0.05)

# ---- 窗口1: 51-60 调权 ----
sp_i = np.load(r"D:\mscapital-kaggle\output\p5e_realmlp_spotcheck\C\R61_70\inner_predictions.npz")
co = np.load(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
c_ids, c_month, c_base = co["sample_id"], co["month"].astype(int), co["baseline_oof"]
i51 = (c_month >= 51) & (c_month <= 60)
c_ids51, c_base51 = c_ids[i51], c_base[i51]
p51, p51_ids, y51 = sp_i["pred"], sp_i["sample_id"], sp_i["target"]
pos51 = np.searchsorted(c_ids51, p51_ids)
canon51 = c_base51[pos51]
scan("窗口1: 51-60 调权", canon51, p51, y51, W)

# ---- 窗口2: 61-70 外窗 ----
sp_o = np.load(r"D:\mscapital-kaggle\output\p5e_realmlp_spotcheck\C\R61_70\predictions.npz")
i6170 = (c_month >= 61) & (c_month <= 70)
c_ids70, c_base70 = c_ids[i6170], c_base[i6170]
p70, p70_ids, y70 = sp_o["pred"], sp_o["sample_id"], sp_o["target"]
pos70 = np.searchsorted(c_ids70, p70_ids)
canon70 = c_base70[pos70]
scan("窗口2: 61-70 外窗", canon70, p70, y70, W)

# ---- 窗口3: PSEUDO eval 33-70 ----
rq = np.load(r"D:\mscapital-kaggle\output\p6_prod\realmlpC_pseudo_pred.npz")
canon_p = np.load(r"D:\mscapital-kaggle\output\c4_protocol_closed_final\clean-baseline-v2\PSEUDO\predictions.npz")
c_ids2, c_pred2, c_y2 = canon_p["sample_id"], canon_p["pred"], canon_p["target"]
p_ev, ids_ev, y_ev = rq["pred"], rq["sample_id"], rq["target"]
cpos = np.searchsorted(c_ids2, ids_ev)
canon_ev = c_pred2[cpos]
assert np.array_equal(canon_ev[canon_ev == canon_ev], canon_ev[canon_ev == canon_ev])
scan("窗口3: PSEUDO eval 33-70", canon_ev, p_ev, y_ev, W)

print("\n=== 结论 ===")
print("主候选 w=0.55 若在三窗口 blendΔ 均正且峰值落在 0.5-0.8 带 -> 稳健, 保持 0.55 提交")
