# B2 — 特征精简 90/84/77

> 阶段: Baseline 表格阶梯 (2026-08-10/11) | 日期: 2026-08-10 | 状态: **GREEN**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
删除 EWM/成交特征是否有损失?

## Hypothesis
EWM/成交可删

## Motivation
B1_ab

## Data
90 features

## Validation Protocol
CV1

## Method
删 EWM(84) / +删成交(77)

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | 0.130204 |
| 实验分数 | 0.130237/0.129744 |
| Delta | -0.0000 |
| Public LB | - |

## Decision
**GREEN**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: C1

## 复现入口
- Scripts: `scripts/06_exp_feature_prune.py`
- Outputs: `-`
- Reports: `RESULTS.md`
