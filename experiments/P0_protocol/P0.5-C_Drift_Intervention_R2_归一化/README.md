# P0.5-C — Drift Intervention (R2 归一化)

> 阶段: P0 Protocol 验证 (2026-08-11) | 日期: 2026-08-11 | 状态: **GREEN**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
归一化表示能否延长 alpha 寿命?

## Hypothesis
R2 归一化替换漂移特征有效

## Motivation
P0-02

## Data
90 features

## Validation Protocol
T3/T4/H2/PSEUDO

## Method
R0(90) vs R1(删top10漂移) vs R2(归一化替换)

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | R0 |
| 实验分数 | R2 4/4 folds 全正 |
| Delta | +0.0023 mean |
| Public LB | - |

## Decision
**GREEN**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: P0.5-D

## 复现入口
- Scripts: `scripts/23_drift_intervention.py`
- Outputs: `-`
- Reports: `RESULTS.md`
