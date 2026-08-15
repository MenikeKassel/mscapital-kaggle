# P1-01e — 特征相对化第二轮

> 阶段: P1 表示与序列 (2026-08-11/12) | 日期: 2026-08-11 | 状态: **RED**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
到达率/强度类特征相对化是否更好?

## Hypothesis
相对化延长 alpha 寿命

## Motivation
P0.5-C 成功

## Data
micro features v2

## Validation Protocol
T4/PSEUDO

## Method
38_micro_rel2

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | P1-01c |
| 实验分数 | T4 +0.0003 / PSEUDO -0.0012 |
| Delta | -0.0012 (PSEUDO) |
| Public LB | - |

## Decision
**RED**

## Failure Analysis
相对化在 PSEUDO 反而更差; 第一轮微观特征已是最优

## Do Not Repeat
不再对微观特征做第二轮相对化

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: (无直接后继 — 见 experiment-index.md)

## 复现入口
- Scripts: `scripts/38_micro_rel2.py`
- Outputs: `-`
- Reports: `RESULTS.md (N006)`
