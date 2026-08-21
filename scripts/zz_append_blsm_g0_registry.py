# -*- coding: utf-8 -*-
"""追加 BLSM-G0 到 registry.csv (28 列, 仿 P6-R 行格式)."""
import csv
from pathlib import Path

row = [
    "BLSM-G0", "canonical",
    "Behavior Existence Gate (行为隐状态存在性)", "blsm-g0",
    "P9_blsm", "2026-08-21", "completed", "EXIST",
    "", "", "BLSM", "behavior|latent|absorption|resiliency",
    "", "", "", "N/A (无训练, 纯统计诊断)",
    "N/A", "PCA/GMM 诊断 + R2/AUC/IC 检验",
    "N/A",
    "行为特征 8PC 累计解释 0.60; 4 代理(order_count/txn_count/vol)解释 PC R2≈0 (−0.0006~−0.0009)",
    "行为 latent 独立于 activity/volatility/month (month AUC≈0.5); PC4 rankIC −0.0036 (p<5e-5)",
    "-",
    "BLSM 行为状态结构存在 (EXIST): 27 特征 PCA↘8 维; 关键证据=现有活动度/波动/计数代理几乎无法解释行为 latent (R2≈0), 不被月主导; 仅 PC4 与 target 显著相关 (IC −0.0036, 量级弱)。行为状态存在但不强 → 进 G1 (incremental gate) 回答是否有基线之上增量",
    "B1/B2 (aggressiveness/persistence) 与 P9-B RED 重叠, 原始统计不重跑 (红线)",
    "原始 iat/burst 事件时距不重跑",
    "scripts/blsm_g0_build.py scripts/blsm_g0_diag.py",
    "docs/blsm-g0-report.md",
    "processed/blsm_g0_train.parquet",
]

path = Path(r"D:\mscapital-kaggle\experiments\registry.csv")
with open(path, "a", encoding="utf-8", newline="") as f:
    csv.writer(f).writerow(row)
print("appended:", open(path, encoding="utf-8-sig").readlines()[-1][:80])
