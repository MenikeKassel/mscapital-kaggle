# -*- coding: utf-8 -*-
"""追加 P9-A-V1 (H14 多 seed 验证) 记录到 registry.csv."""
import csv

rows = [
    ["P9-A-V1","canonical","P9-A Cancel 3-seed robustness + mask 机制验证","p9-a-h14-v1","p9_lite","2026-08-20","completed","GREEN","P9-A-LITE","C-05","",
     "cancel|liquidity-withdrawal|robustness|seed","","","","realmlp(3x256) C-05 E0","uncentered cosine (frozen 51-70)","3 seed (2026/2027/2028) x base/raw/mask",
     "0.114364 (base 2026)","raw mean +0.007148","raw 3-seed +0.004146/+0.005986/+0.011313 (3/3 正)","",
     "撤单侧拆不对称稳健 GREEN 候选: 3-seed 均值 frozen +0.0071 (3/3 正); 单 seed 2026 的 low_act 拖累 (-0.0064) 是 seed 噪声 (2027/28 low_act +0.0105/+0.0073), 双 regime 均值皆正; mask(低撤单置0) 三 seed 均劣于 raw → 机制猜想证伪, hard-mask 路径不重复; 与 Z 同源 (Z_ob_cancel_side_imb) 互为归因, 是 Z 绿灯候选主成分","hard-mask 低撤单样本不能修复 regime 集中 (真信号低活动也有, 清空=丢信号)","","scripts/p9_lite_train.py","docs/p9-lite-results.md","output/p9_lite/pkgA/{feat_s*,feat_*_mask}"],
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
print("appended; last row:", "P9-A-V1" in "".join(lines))
