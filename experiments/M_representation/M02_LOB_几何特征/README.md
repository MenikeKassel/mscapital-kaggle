# M02 — LOB 几何特征

> 阶段: M 系列残差表示 (2026-08-13) | 日期: 2026-08-13 | 状态: **RED**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
盘口几何 (深度/形状) 有无残差 alpha?

## Hypothesis
几何特征有信息

## Motivation
表示探索

## Data
market LOB

## Validation Protocol
canonical residual OOF

## Method
lob_geometry 特征 → 残差学习

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | canonical |
| 实验分数 | ~0 |
| Delta | ~0 |
| Public LB | - |

## Decision
**RED**

## Failure Analysis
几何特征无残差增量

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: (无直接后继 — 见 experiment-index.md)

## 复现入口
- Scripts: `scripts/m02_* (src/mscapital/features/lob_geometry.py)`
- Outputs: `output/m02_* output/m02t_*`
- Reports: `docs/m02-geometry-results.md`
