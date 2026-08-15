# P2-cal — RealMLP PSEUDO 定标 + 尺度门禁

> 阶段: P2 校准 (2026-08-12) | 日期: 2026-08-12 | 状态: **GREEN**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
v7 融合的 PSEUDO/LB gap 与组件尺度迁移?

## Hypothesis
Regime B gap 0.0047; 尺度门禁 test/valid=0.74

## Motivation
v8 后校准接手

## Data
152 features + table OOF

## Validation Protocol
PSEUDO (m0-32/m33-70)

## Method
RealMLP PSEUDO 训练 + v5 表格重放 + 尺度诊断

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | v5 table PSEUDO 0.134871 |
| 实验分数 | RealMLP 0.138560; v7 融合 0.139683 |
| Delta | +0.0048 |
| Public LB | v7 LB 0.135 |

## Decision
**GREEN**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: Protocol-v2 C 系列

## 复现入口
- Scripts: `scripts/kaggle_realmlp_pseudo/ (kernel)`
- Outputs: `output/rlps_v12/`
- Reports: `RESULTS.md §P2`
