# P5-E — RealMLP 生产 learner spot-check

> 阶段: P5 市场探针 (2026-08-14/15) | 日期: 2026-08-15 | 状态: **GREEN**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
SCFI 在生产 learner (RealMLP) 上是否兑现?

## Hypothesis
standalone +0.0040 / blend +0.0014

## Motivation
P5-B 的 NN 空白

## Data
152+Z features

## Validation Protocol
R61_70 (inner 0-50/tune 51-60/refit 0-60/outer 61-70)

## Method
RealMLP(152+Z) vs (152)

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | A 臂 0.148526 |
| 实验分数 | C 0.152570 (Δ+0.004044); blend w=0.50 Δ+0.001369; corr 0.940 |
| Delta | +0.004044 |
| Public LB | - |

## Decision
**GREEN**

## Failure Analysis
SmallMLP 代理被推翻 (过弱)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: P6 生产推理

## 复现入口
- Scripts: `scripts/p5e_prep_realmlp.py scripts/p5e_realmlp_spotcheck.py scripts/p5e_blend_check.py`
- Outputs: `output/p5e_realmlp_spotcheck`
- Reports: `docs/p5de-production-verification.md`
