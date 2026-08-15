# P4-MH — Market history 特征

> 阶段: P4 隐藏信息调查 (2026-08-13/14) | 日期: 2026-08-14 | 状态: **YELLOW**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
600s market 历史聚合特征是否有效?

## Hypothesis
有独立信息 (H4 直接变现)

## Motivation
P4-01a

## Data
market 600s

## Validation Protocol
multi-outer

## Method
历史特征 → 残差学习

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | canonical |
| 实验分数 | 弱正 |
| Delta | ~+0.0005 |
| Public LB | - |

## Decision
**YELLOW**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: P5-01 序列版

## 复现入口
- Scripts: `scripts/p4_market_history.py`
- Outputs: `output/p4_market_history_*`
- Reports: `docs/p4-hidden-information-report.md`
