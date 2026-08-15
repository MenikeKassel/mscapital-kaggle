# P4-LB142 — LB142 分歧取证

> 阶段: P4 隐藏信息调查 (2026-08-13/14) | 日期: 2026-08-14 | 状态: **YELLOW**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
外部 LB142 与我们差异最大的样本有何特征?

## Hypothesis
分歧样本揭示外部模型信息面

## Motivation
LB 逆向

## Data
152 features + lb142 pred

## Validation Protocol
诊断

## Method
分歧样本分析

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | - |
| 实验分数 | 分歧集中于高波动/低流动样本 |
| Delta | - |
| Public LB | - |

## Decision
**YELLOW**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: P5 探针

## 复现入口
- Scripts: `scripts/p4_lb142_forensics.py`
- Outputs: `output/p4_lb142_forensics`
- Reports: `docs/p4-hidden-information-report.md`
