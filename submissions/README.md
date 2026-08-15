# Submissions 登记表

> 可追溯性: 任意 LB 分数 → 生成脚本 → 实验 ID。
> 物理文件: `output/submissions/*.csv` (gitignore, 不入 git; 本表为权威登记)。
> 纪律: 新提交必须先登记本表再提交 Kaggle。
> 更新: 2026-08-15 (仓库工程化整理 Phase L)

## 正式提交历史 (有 LB 记录)

| Submission | 实验 | 配方 | Local (PSEUDO) | Public LB | 日期 | 生成脚本 |
| -- | -- | -- | -: | -: | -- | -- |
| v1 | G3 | LGB+XGB+MLP-ens 0.1/0.4/0.5 | 0.138931 | **0.122** (#79) | 08-11 00:11 | scripts/16_final_submission.py |
| v2 | H1 | XGB+Cat+MLP 0.1/0.4/0.5 | 0.139332 | **0.122** | 08-11 08:46 | scripts/18_final_submission_v2.py |
| v3 | P0-03 | temporal 权重重估 | 0.130015 | **0.122** | 08-11 10:43 | scripts/22_final_submission_v3.py |
| v4 (SUB-v4) | P0.5-D | R2 归一化 + temporal | +0.0015 | **0.123** ✅ | 08-11 12:08 | scripts/26_final_submission_v4.py |
| v5 (SUB-v5) | P1-01c | R2 + 22 微观融合 | 0.134871 | **0.125** ✅ | 08-12 00:15 | scripts/35_final_v5.py |
| v6 | P1-02 | 表格 + TCN 0.07 | +0.004 (TCN) | **0.082** ❌ N005 | 08-12 01:38 | scripts/37_final_v6.py |
| v7 (SUB-v7) | SUB-v7 | v5 + RealMLP 0.8/0.2 | 0.139683 | **0.135** ✅ | 08-12 10:00 | scripts/42_realmlp_fusion.py |
| v7b (SUB-v7) | SUB-v7 | 0.75/0.25 | — | 0.134 | 08-12 | scripts/42_realmlp_fusion.py |
| v8a (SUB-v8) | SUB-v8 | v7 + lb142 0.7/0.3 | — | 0.139 | 08-12 13:24 | scripts/43_lb142_fusion.py |
| **v8b (SUB-v8)** | SUB-v8 | v7 + lb142 **0.5/0.5** | — | **0.142** (#30) ✅ | 08-12 13:24 | scripts/43_lb142_fusion.py |
| (realmlp 单模) | SUB-v7 | RealMLP 单模型 | 0.1439 | 0.134 | 08-12 09:59 | scripts/41_realmlp_local.py |

**提交轨迹**: 0.122 → 0.123 → 0.125 → 0.135 → 0.142; 排名 #82 → #48 → #30
**校准纪律**: 本地过 PSEUDO 门禁才占用提交配额; 剩余配额留作最终提交。

## 磁盘文件 ↔ 登记对照

| 文件 (output/submissions/) | 对应条目 | 说明 |
| -- | -- | -- |
| submission_blend_v1~v6.csv | v1~v6 | 正式提交文件 |
| submission_blend_v7_rl15/20/25.csv | v7/v7b 候选 | 权重 0.15/0.20/0.25, 0.20 为甜点位 |
| submission_v8_ref30~70.csv | v8 候选 | lb142 权重 0.3~0.7, 0.5 为甜点位 |
| b1_official_lgb.csv | B1 | 官方基线复刻 |
| realmlp_submission.csv | RealMLP 单模 | 08-12 提交 |
| submission_cbv2_calib.csv | C4 系 | clean baseline v2 校准版 (08-13) |
| submission_v9_t5r30~t10r50.csv (6) | SUB-v8 后 | t5/t10 特征 × ref30/40/50 — **无 LB 记录** ⚠️ |
| submission_v9_cos_a10~a25.csv (4) | SUB-v8 后 | cosine α 变体 — **无 LB 记录** ⚠️ |
| submission_v9_cos_simple_a10/13.csv (2) | SUB-v8 后 | cosine simple 变体 — **无 LB 记录** ⚠️ |
| submission_v9_cos_solo.csv (1) | SUB-v8 后 | cosine solo — **无 LB 记录** ⚠️ |

> ⚠️ v9 系列 (13 个文件, 08-12/14) 未登记 LB — 若提交过, 请补充数值到本表 + RESULTS.md。
> 若从未提交, 建议归档到 `output/submissions/_never_submitted/` 防止误用。

## P6 提交候选 (未提交, 等拍板)

| 候选 | 配方 | 门禁证据 | 期望 LB |
| -- | -- | -- | -- |
| submission_candidate_p6.csv | v8b + 0.55×RealMLP(152+Z) | 双窗口 blendΔ +0.0014 一致; std 比 0.9726; corr 结构吻合 | 0.1423 ~ 0.1428 |

生成: `output/p6_prod/` (scripts/p6_prod_realmlp.py → p6_finish_test.py); 报告 docs/p6-production-inference.md

## LB 快照存档

Leaderboard 历史快照: `_archive/lb_snapshots/` (7 份, 08-11 ~ 08-12, 含排名变化)
