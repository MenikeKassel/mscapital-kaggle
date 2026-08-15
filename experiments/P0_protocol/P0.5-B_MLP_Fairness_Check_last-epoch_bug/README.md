# P0.5-B — MLP Fairness Check (last-epoch bug)

> 阶段: P0 Protocol 验证 (2026-08-11) | 日期: 2026-08-11 | 状态: **SUPERSEDED**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
Temporal Matrix 对 MLP 是否不公平?

## Hypothesis
last-epoch 评估低估 MLP

## Motivation
P0-01 复盘

## Data
90 features

## Validation Protocol
T1-T3

## Method
30ep/60ep best-state 重评估

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | T1 0.11062 |
| 实验分数 | T1 +0.0062 / T2 +0.0045 / T3 +0.0061 |
| Delta | +0.0050 平均 |
| Public LB | - |

## Decision
**SUPERSEDED**

## Failure Analysis
评估协议高估其劣势; 修正后 MLP 仍弱于 CatBoost (T2 差 0.0054)

## Do Not Repeat
评估模型必须用 best-state 而非 last-epoch

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: (无直接后继 — 见 experiment-index.md)

## 复现入口
- Scripts: `scripts/24_mlp_fairness.py`
- Outputs: `-`
- Reports: `RESULTS.md`
