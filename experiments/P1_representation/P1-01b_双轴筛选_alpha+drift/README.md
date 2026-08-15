# P1-01b — 双轴筛选 (alpha+drift)

> 阶段: P1 表示与序列 (2026-08-11/12) | 日期: 2026-08-11 | 状态: **YELLOW**
> Alias (历史编号): P1-1b (历史简写)
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
新特征 alpha 轴与漂移轴表现?

## Hypothesis
alpha 正但部分携带漂移

## Motivation
P1-01a

## Data
micro features

## Validation Protocol
T3/T4/PSEUDO

## Method
CatBoost 双轴审计

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | 152 |
| 实验分数 | alpha mean +0.0022 |
| Delta | +0.0022 |
| Public LB | - |

## Decision
**YELLOW**

## Failure Analysis
drift 轴 ΔAUC +0.0168 ⚠️ 需第二轮归一化; 净效果为正

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: P1-01c

## 复现入口
- Scripts: `scripts/28_feature_screening.py`
- Outputs: `-`
- Reports: `RESULTS.md`
