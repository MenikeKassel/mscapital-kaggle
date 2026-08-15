# E02 — Reconditionor-lite

> 阶段: E 系列状态条件化 (2026-08-13) | 日期: 2026-08-13 | 状态: **RED**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
状态条件创新 (O−E[O|M]) 能否预测残差?

## Hypothesis
窗口内 residual cosine 0.013

## Motivation
条件创新方向

## Data
order + market 状态

## Validation Protocol
canonical residual OOF

## Method
条件化残差学习

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | canonical |
| 实验分数 | 窗口内 cos 0.013, 跨月不可变现 |
| Delta | ~0 |
| Public LB | - |

## Decision
**RED**

## Failure Analysis
窗口内可解释 ≠ 跨月可预测增量 (负面链 #2)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: P5-B SCFI (严格版)

## 复现入口
- Scripts: `scripts/e02_* (src/mscapital/models/context_shift.py)`
- Outputs: `output/e02_*`
- Reports: `docs/e01-e02-e03-results.md`
