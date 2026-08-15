# P6R-00 — Retrieval 残差检索

## Metadata

- ID: `P6R-00` (canonical)
- Phase: P6_production | Created: 2026-08-14
- Status: `completed` | Decision: **RED**
- Parent: P4-08 | Successor: P6R-01
- Aliases: -
- Tags: retrieval|negative

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | RMS frozen |
| Score | 32/32 候选全正, 最大 PSEUDO Δ=+0.000588 (gate 39%) |
| Delta | +0.000588 |
| Public LB | - |

## Decision

**RED** (completed)

## Failure Analysis

无候选同时满足 CI 下界>0 与 α≥0.10; 相似状态残差均值预测力弱; 负面链 #5

## Do Not Repeat

不再做检索-残差预测路线

## 复现入口

- Scripts: `scripts/p6r_00_retrieval_residual.py scripts/p6r_00_variants.py`
- Reports: `docs/p6r_experiment_report.md`
- Artifacts: `output/p6r_00`
