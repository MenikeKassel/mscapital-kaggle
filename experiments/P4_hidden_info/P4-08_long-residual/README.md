# P4-08 — Long residual (短/长/双窗)

## Metadata

- ID: `P4-08` (canonical)
- Phase: P4_hidden | Created: 2026-08-14
- Status: `completed` | Decision: **RED**
- Parent: P4-01 | Successor: P6R-00
- Aliases: P4-06A
- Tags: residual|negative

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | canonical |
| Score | +0.0000 |
| Delta | +0.0000 |
| Public LB | - |

## Decision

**RED** (completed)

## Failure Analysis

聚合残差无预测增量 (负面链 #1)

## Do Not Repeat

不做残差均值直接预测

## 复现入口

- Scripts: `scripts/p4_06a_long_residual.py`
- Reports: `docs/p4-hidden-information-report.md`
- Artifacts: `output/p4_06a*`
