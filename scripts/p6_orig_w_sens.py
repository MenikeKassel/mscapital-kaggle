# -*- coding: utf-8 -*-
"""原创候选 w 敏感性: v5表格 + RealMLP-C, 扩网格到 w=2.0, 跨窗口确认.
窗口1: PSEUDO eval 33-70 (已有 v5_table_pseudo + realmlpC_pseudo)
窗口2: 61-70 (RealMLP-C R61_70 + canonical OOF as表格代理? 不, 用 v5表格在 61-70)
  注意: v5表格无独立 61-70 OOF 文件; 用 canonical PSEUDO 的 month 字段切 61-70
  作为表格侧代理有偏差。因此跨窗口一致性改用: PSEUDO eval 33-70 内部分期对比
  (33-50 vs 51-70) 看 w 峰是否稳定。
"""
import numpy as np

def cosine(p, y):
    return float(np.dot(p, y) / np.sqrt(np.dot(p, p) * np.dot(y, y)))

def rms_norm(p):
    s = float(np.sqrt(np.mean(p ** 2)))
    return p / s if s > 0 else p

tbl = np.load(r"D:\mscapital-kaggle\output\rlps_v12\v5_table_pseudo_pred.npz")
rlc = np.load(r"D:\mscapital-kaggle\output\p6_prod\realmlpC_pseudo_pred.npz")
pt, y = tbl["pred"], tbl["y"].astype(np.float64)
pc = rlc["pred"]
# canonical 有 month, 用它切 61-70 与 33-50 做分期一致性
canon = np.load(r"D:\mscapital-kaggle\output\c4_protocol_closed_final\clean-baseline-v2\PSEUDO\predictions.npz")
cids = canon["sample_id"]
# rlc 有 sample_id
rids = rlc["sample_id"]
# 用 sample_id 建 index
ridx = {int(i): k for k, i in enumerate(rids)}
# canonical month 对应行 -> 映射到 rlc 的索引
c_month = canon["month"].astype(int)
# tilde: 假设 v5/rlc 顺序 == cids 顺序 (已验证三目标同序)
months = c_month
W = np.arange(0.05, 2.01, 0.05)

def scan(sel, label):
    m = months[sel]
    y_sel = y[sel]
    p1 = pt[sel]
    p2 = pc[sel]
    cb = cosine(p1, y_sel)
    best_w, best_d, d20 = None, -1e9, None
    for w in W:
        pb = rms_norm(p1) + w * rms_norm(p2)
        d = cosine(pb, y_sel) - cb
        if d > best_d:
            best_d, best_w = d, float(w)
    d20 = cosine(rms_norm(p1) + 0.20 * rms_norm(p2), y_sel) - cb
    d60 = cosine(rms_norm(p1) + 0.60 * rms_norm(p2), y_sel) - cb
    print(f"{label}: base={cb:.6f} w=0.20 Δ={d20:+.6f} w=0.60 Δ={d60:+.6f} best w={best_w:.2f} Δ={best_d:+.6f}")
    return best_w, best_d

print("PSEUDO eval 33-70 分期一致性 (v5表格 + w*RealMLP-C):")
scan(np.arange(len(y)) >= 0, "  全期 33-70 ")
scan((months >= 33) & (months <= 50), "  分1 33-50 ")
scan((months >= 51) & (months <= 70), "  分2 51-70 ")
