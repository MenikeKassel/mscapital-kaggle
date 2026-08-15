# P4-04 — Disagreement forensics + 0.5 价簇

> 阶段: P4 隐藏信息调查 (2026-08-13/14) | 日期: 2026-08-14 | 状态: **GREEN**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
0.5 价格群是第二 instrument 还是计算假象?

## Hypothesis
H1' 第二 instrument

## Motivation
LB 分析

## Data
market

## Validation Protocol
诊断

## Method
价簇取证

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | - |
| 实验分数 | ask 全空 → mid=0.5 计算假象 |
| Delta | - |
| Public LB | - |

## Decision
**GREEN**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: (2026-08-15 示例报告确认)

## 复现入口
- Scripts: `scripts/p4_04_disagreement.py`
- Outputs: `output/p4_04_* output/p4_lb142_forensics`
- Reports: `docs/p4-hidden-information-report.md`
