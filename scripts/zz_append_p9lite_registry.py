# -*- coding: utf-8 -*-
"""追加 3 条 P9-Lite 记录到 registry.csv (csv 模块安全引用)."""
import csv

rows = [
    ["P9-A-LITE","canonical","P9-Lite Cancel Pressure gate probe","p9-lite-cancel","p9_lite","2026-08-20","completed","YELLOW","C-05","","",
     "cancel|liquidity-withdrawal|order","","","","realmlp(3x256) C-05 E0","uncentered cosine (frozen 51-70)","152 -> +13 side-split cancel",
     "0.114364 (base frozen)","0.118510 (feat frozen)","+0.004146 frozen / +0.004378 eval","",
     "侧拆撤单不对称(流动性撤退)真实有效: +0.0041 frozen / 20-20 聚合月正 / 13-20 月Δ正, 但 regime 集中 (hi_act +0.0103 / low_act -0.0064) → 非 clean GREEN, 进联合/校准验证; 与 Z 绿灯同源 (Z_ob_cancel_side_imb) 互为归因","","",
     "scripts/p9_lite_train.py","docs/p9-lite-results.md","output/p9_lite/pkgA"],
    ["P9-B-LITE","canonical","P9-Lite Event-Time gate probe","p9-lite-eventtime","p9_lite","2026-08-20","completed","RED","C-05","","",
     "event-time|iat|burst","","","","realmlp(3x256) C-05 E0","uncentered cosine (frozen 51-70)","152 -> +18 raw iat/burst",
     "0.114364 (base frozen)","0.110477 (feat frozen)","-0.003887 frozen / -0.002754 eval","",
     "原始 iat/burst 聚合负增益; 152 基线已含事件节奏 (o_*_near_far / t_*_gap / rowcount_near_far), 未条件化原始聚合为冗余噪声变体; 事件节奏信息存在须经 Z 式 market/tx 条件化 (Z 绿灯), 原始形式无增量",
     "compressed tabular baseline 已包含事件节奏信息; 原始聚合叠加有害","",
     "scripts/p9_lite_train.py","docs/p9-lite-results.md","output/p9_lite/pkgB"],
    ["P9-C-LITE","canonical","P9-Lite M55-lite gate probe","p9-lite-m55","p9_lite","2026-08-20","completed","YELLOW","C-05","","",
     "dwi|entropy|depth-imbalance","","","","realmlp(3x256) C-05 E0","uncentered cosine (frozen 51-70)","152 -> +10 M55-lite",
     "0.114364 (base frozen)","0.114906 (feat frozen)","+0.000542 frozen / +0.000704 eval","",
     "L1/L2 DWI + trade entropy 边缘增 +0.0005~+0.0007, 方向稳定 12/20 月Δ正, 略偏 hi_act → YELLOW 进联合实验; DWOFI 仅 L1/L2 两层可构造, 熵与 t_buy_ratio 重叠, 与 P10-FM M1 L2 档位 (+0.0008 边缘) 一致","","",
     "scripts/p9_lite_train.py","docs/p9-lite-results.md","output/p9_lite/pkgC"],
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
print("appended 3 rows; total data rows =", len([l for l in lines if l.strip()]) + 3 - 1)
