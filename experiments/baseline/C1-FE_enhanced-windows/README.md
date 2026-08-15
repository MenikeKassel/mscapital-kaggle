# C1-FE — 增强窗口特征 90→109

## Metadata

- ID: `C1-FE` (legacy)
- Phase: baseline | Created: 2026-08-10
- Status: `completed` | Decision: **RED**
- Parent: B2 | Successor: -
- Aliases: -
- Tags: features

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | 0.130204 |
| Score | 0.129760 |
| Delta | -0.0004 |
| Public LB | - |

## Decision

**RED** (completed)

## Failure Analysis

特征工程到顶, 90 特征即甜点位

## Do Not Repeat

不再盲目堆窗口统计特征

## 复现入口

- Scripts: `scripts/07_exp_enhanced_windows.py`
- Reports: `RESULTS.md`
- Artifacts: `-`
