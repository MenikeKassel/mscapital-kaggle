# P5-01 — Market-only 序列审判

> 阶段: P5 市场探针 (2026-08-14/15) | 日期: 2026-08-14 | 状态: **YELLOW**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
market 600s 序列 (非聚合) 是否含独立方向信息?

## Hypothesis
含 (H4 变现)

## Motivation
P4-01a

## Data
market 600s 序列

## Validation Protocol
canonical OOF

## Method
market 序列编码 → 残差学习

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
序列信息大部分已在聚合特征中

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: P5-02I 信息定位

## 复现入口
- Scripts: `scripts/p5_01_market_sequence.py`
- Outputs: `output/p5_01_market_sequence`
- Reports: `docs/p5-02i-info-audit-report.md`
