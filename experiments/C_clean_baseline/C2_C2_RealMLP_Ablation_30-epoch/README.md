# C2 — C2 RealMLP Ablation (30-epoch)

> 阶段: C 系列 Clean Baseline v2 (2026-08-13) | 日期: 2026-08-13 | 状态: **GREEN**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
RealMLP 训练方案消融: 30ep vs 全量?

## Hypothesis
30ep 是最优生产方案

## Motivation
C1

## Data
152 features

## Validation Protocol
inner 筛选 (P100) + formal outer

## Method
mask-full/mask-none/optimizer-first-ntp/target-raw/ceiling-30 消融

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | C1 |
| 实验分数 | 30ep 全 outer 最优 |
| Delta | + |
| Public LB | - |

## Decision
**GREEN**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: C3

## 复现入口
- Scripts: `scripts/kaggle_c2_*`
- Outputs: `output/c2_* output/kaggle_c2_*`
- Reports: `docs/c2-realmlp-ablation*.md`
