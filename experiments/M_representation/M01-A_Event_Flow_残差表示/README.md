# M01-A — Event Flow 残差表示

> 阶段: M 系列残差表示 (2026-08-13) | 日期: 2026-08-13 | 状态: **RED**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
事件流聚合残差能否预测 OOF 残差?

## Hypothesis
event-flow 有残差 alpha

## Motivation
残差研究链

## Data
order/tx → residual OOF

## Validation Protocol
canonical residual OOF

## Method
event flow 特征 → 残差 CatBoost

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | canonical |
| 实验分数 | ~0 |
| Delta | ~0 |
| Public LB | - |

## Decision
**RED**

## Failure Analysis
事件流特征无残差增量

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: (无直接后继 — 见 experiment-index.md)

## 复现入口
- Scripts: `scripts/m01a_* (src/mscapital/models/m01a.py)`
- Outputs: `output/m01a_*`
- Reports: `docs/m01-a-results.md`
