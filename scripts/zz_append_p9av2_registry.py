# -*- coding: utf-8 -*-
"""追加 P9-A-V2 (三方对比) 记录到 registry.csv."""
import csv

rows = [
    ["P9-A-V2","canonical","三方对比: cancel x Z (替代/叠加裁决)","p9-a-v2-3way","p9_lite","2026-08-20","completed","RED","P9-A-V1","","",
     "cancel|Z|attribution|ensemble","","","","realmlp(3x256) C-05 E0","uncentered cosine (frozen 51-70)","seed 2026 x base/cancel/Z/cancel+Z",
     "0.114364 (base frozen)","Z 0.123030 (+0.008666)","cancel +0.004146; Z +0.008666; cancel+Z +0.007852 (J−Z −0.0008)","",
     "撤单族归因闭合: cancel ⊂ Z — 替代否决 (A 只有 Z 一半强), 叠加否决 (J−Z −0.0008, 月度Δ正 10/20); Z 含 market/tx 条件化的 Z_ob_cancel_side_imb, 撤单族是 Z 绿灯的驱动成分但 Z 表达更好; 结论「直接用 A」不成立, 撤单族不做独立生产特征, 生产资产仍 152+73Z; A 的价值 = 归因证据 + Z 可解释成分","raw 撤单侧拆叠加到 152+Z 无增量 (信息已被 Z 覆盖)","","scripts/p9_lite_joint_build.py","docs/p9-lite-results.md","output/p9_lite/{pkgJ,pkgA/feat_z,pkgA/feat_j}"],
]

path = r"D:\mscapital-kaggle\experiments\registry.csv"
with open(path, "r", newline="", encoding="utf-8") as f:
    lines = f.readlines()
while lines and lines[-1].strip() == "":
    lines.pop()
with open(path, "w", newline="", encoding="utf-8") as f:
    f.writelines(lines)
    w = csv.writer(f)
    for r in rows:
        w.writerow(r)
print("ok")
