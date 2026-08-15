# M05 — Market-State KNN

> 阶段: M 系列残差表示 (2026-08-13) | 日期: 2026-08-13 | 状态: **RED**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
历史相似市场状态 KNN 有无预测力?

## Hypothesis
状态检索有残差 alpha

## Motivation
状态条件化雏形

## Data
market 状态

## Validation Protocol
canonical residual OOF

## Method
状态 KNN → 残差均值

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
状态检索无残差增量 (P6R 前身)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: P6R

## 复现入口
- Scripts: `scripts/m05_* (src/mscapital/models/m05.py)`
- Outputs: `output/m05_*`
- Reports: `docs/m05-market-state-results.md`
