# M04 — Optiver Interaction 特征族

> 阶段: M 系列残差表示 (2026-08-13) | 日期: 2026-08-13 | 状态: **RED**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
Optiver 风格交互特征有无残差 alpha?

## Hypothesis
交互特征有信息

## Motivation
类似比赛迁移 (TATC)

## Data
order/tx/market

## Validation Protocol
canonical residual OOF

## Method
optiver_interactions 特征 → 残差学习

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
交互特征无残差增量

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: (无直接后继 — 见 experiment-index.md)

## 复现入口
- Scripts: `scripts/m04_* (src/mscapital/features/optiver_interactions.py)`
- Outputs: `output/m04_*`
- Reports: `docs/m04-optiver-interactions-results.md`
