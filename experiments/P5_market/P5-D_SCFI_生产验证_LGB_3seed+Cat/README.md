# P5-D — SCFI 生产验证 (LGB 3seed+Cat)

> 阶段: P5 市场探针 (2026-08-14/15) | 日期: 2026-08-15 | 状态: **GREEN**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
SCFI 在 3-seed 融合上是否跨块稳定?

## Hypothesis
C 臂双块正且最强

## Motivation
P5-B 升级裁决

## Data
152+Z features

## Validation Protocol
双块 (B1/B2), holdout 调权

## Method
LGB 3-seed + CatBoost 臂对比

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | 152-only |
| 实验分数 | C/LGB +0.000849 (B1 +0.0013/B2 +0.0004) |
| Delta | +0.000849 |
| Public LB | - |

## Decision
**GREEN**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: P5-E

## 复现入口
- Scripts: `scripts/p5d_prod_blend.py`
- Outputs: `output/p5d_prod_blend`
- Reports: `docs/p5de-production-verification.md`
