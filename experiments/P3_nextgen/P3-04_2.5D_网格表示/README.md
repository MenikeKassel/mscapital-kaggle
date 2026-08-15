# P3-04 — 2.5D 网格表示

> 阶段: P3 下一代方法 (2026-08-14) | 日期: 2026-08-14 | 状态: **RED**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
LOB 投影网格有无信号?

## Hypothesis
网格保留空间结构

## Motivation
外部方法 (CSM-Net 类)

## Data
LOB 网格

## Validation Protocol
PSEUDO/H2/T3/T4

## Method
2.5D 投影 → 小网络

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | canonical |
| 实验分数 | PSEUDO +0.0000 |
| Delta | +0.0000 |
| Public LB | - |

## Decision
**RED**

## Failure Analysis
投影无信号 (信息已在聚合特征中)

## Do Not Repeat
不再做网格投影

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: (无直接后继 — 见 experiment-index.md)

## 复现入口
- Scripts: `scripts/p3_04_grid.py scripts/p3_grid_common.py`
- Outputs: `output/p3_04_*`
- Reports: `docs/p3-results.md`
