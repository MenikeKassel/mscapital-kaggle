# P4-06A — Long residual (短/长/双窗)

> 阶段: P4 隐藏信息调查 (2026-08-13/14) | 日期: 2026-08-14 | 状态: **RED**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
OOF 残差的可解释性可否跨窗复用?

## Hypothesis
残差有跨月结构

## Motivation
残差研究链

## Data
residual OOF

## Validation Protocol
short/long/both 三形态

## Method
长残差聚合

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | canonical |
| 实验分数 | +0.0000 |
| Delta | +0.0000 |
| Public LB | - |

## Decision
**RED**

## Failure Analysis
聚合残差无预测增量 (负面链 #1)

## Do Not Repeat
不做残差均值直接预测

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: P6R

## 复现入口
- Scripts: `scripts/p4_06a_long_residual.py`
- Outputs: `output/p4_06a*`
- Reports: `docs/p4-hidden-information-report.md`
