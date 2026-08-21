# -*- coding: utf-8 -*-
"""原创锚验证: v5表格(纯原创) 基础上叠 RealMLP(152) vs RealMLP-C(152+73Z).
全部用已有 PSEUDO eval 33-70 预测, 不重训. 目标回答:
  今天的 Z 线思路 (152+73Z 作为生产新增量) 在纯原创锚(无 lb142)上是否成立.
对照: v7 原始构成 = v5表格 + RealMLP(152) 0.8/0.2.
"""
import numpy as np

def cosine(p, y):
    return float(np.dot(p, y) / np.sqrt(np.dot(p, p) * np.dot(y, y)))

def rms_norm(p):
    s = float(np.sqrt(np.mean(p ** 2)))
    return p / s if s > 0 else p

# ---- 载入 ----
tbl = np.load(r"D:\mscapital-kaggle\output\rlps_v12\v5_table_pseudo_pred.npz")
rmlp152 = np.load(r"D:\mscapital-kaggle\output\rlps_v12\realmlp_pseudo_pred.npz")
# RealMLP-C 用 sample_id 对齐到 rmlp152 的 y
rmlpC = np.load(r"D:\mscapital-kaggle\output\p6_prod\realmlpC_pseudo_pred.npz")

p_tbl, y_tbl = tbl["pred"], tbl["y"]
p_152, y_152 = rmlp152["pred"], rmlp152["y"]
p_C, idsC, yC = rmlpC["pred"], rmlpC["sample_id"], rmlpC["target"]

# 对齐校验: 三者 y 是否同一集合
assert len(y_tbl) == len(y_152) == len(p_C), "length mismatch"
# 顺序对齐: rmlp152 与 tbl 是否同序; rmlpC 按 sample_id, 与其 y 内部自洽
# 用 y 序列匹配 rmlpC 到 tbl 顺序 (值浮点可能存在 rounding, 用 argsort 对齐)
# 更稳妥: 检查 rmlpC 的 target 是否与 y_tbl 全等(可能顺序不同)
def align_to_ref(p_src, y_src, y_ref):
    """按 y 值序列对齐 (若同集合但不同序). 用 argsort 匹配."""
    s_ref = np.argsort(y_ref, kind="stable")
    s_src = np.argsort(y_src, kind="stable")
    if len(s_ref) != len(s_src):
        raise ValueError("length mismatch")
    # 检查同序 (值差异忽略) -> 若排序后一致则直接用, 否则按排序映射
    if np.allclose(y_src[s_src], y_ref[s_ref], atol=1e-9):
        inv = np.empty_like(s_ref)
        inv[s_src] = np.arange(len(s_src))
        return p_src[inv]
    raise AssertionError("target sets differ — cannot align")

# 直接检查是否已同序
same = np.allclose(y_152, y_tbl, atol=1e-9)
print(f"rmlp152 与 tbl 同序: {same}")
if not same:
    p_152 = align_to_ref(p_152, y_152, y_tbl)
# rmlpC target -> tbl 序
if not np.allclose(yC, y_tbl, atol=1e-9):
    p_C = align_to_ref(p_C, yC, y_tbl)
    print("rmlpC target 与 tbl 同序: False -> aligned")
else:
    print("rmlpC target 与 tbl 同序: True")

y = y_tbl
cos_tbl = cosine(p_tbl, y)
cos_152 = cosine(p_152, y)
cos_C = cosine(p_C, y)
print(f"\nPSEUDO eval 33-70 单模型:")
print(f"  v5表格(纯原创锚)      = {cos_tbl:.6f}")
print(f"  RealMLP(152)          = {cos_152:.6f}")
print(f"  RealMLP-C(152+73Z)    = {cos_C:.6f}")

def scan(label, base, add, wgrid=np.arange(0.05, 0.61, 0.05)):
    cb = cosine(base, y)
    best_w, best_d = None, -1e9
    for w in wgrid:
        pb = rms_norm(base) + w * rms_norm(add)
        d = cosine(pb, y) - cb
        if d > best_d:
            best_d, best_w = d, float(w)
    d20 = cosine(rms_norm(base) + 0.20 * rms_norm(add), y) - cb
    print(f"  {label}: w=0.20 Δ={d20:+.6f} | best w={best_w:.2f} Δ={best_d:+.6f}")
    return best_w, best_d, d20

print("\n--- 纯原创锚 (v5表格 + RealMLP 族, PSEUDO eval 33-70) ---")
_, _, d152_20 = scan("v5tbl+RealMLP(152)      [=v7式 0.8/0.2]", p_tbl, p_152)
_, _, dC_20 = scan("v5tbl+RealMLP-C(52+73Z) [=今天思路纯原创]", p_tbl, p_C)
print(f"\n对比: Z线(RealMLP-C) 相对 RealMLP(152) 的 w=0.20 收益差 = {dC_20 - d152_20:+.6f}")
print("=> 若为正: 今天思路(152+73Z)在纯原创锚上成立, 且不受 lb142 干扰")
