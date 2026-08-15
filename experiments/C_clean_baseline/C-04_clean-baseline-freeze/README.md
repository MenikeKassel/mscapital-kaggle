# C-04 — C4 Clean Baseline v2 冻结

## Metadata

- ID: `C-04` (canonical)
- Phase: C_clean | Created: 2026-08-13
- Status: `completed` | Decision: **GREEN**
- Parent: C-02 | Successor: E-01|M-01|P3-01|P3-02|P3-03|P3-04|P3-05|P4-01
- Aliases: -
- Tags: protocol|baseline

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | RealMLP 单模 |
| Score | 0.142649/0.141762/0.143515/0.156924 |
| Delta | +0.005026 mean |
| Public LB | - |

## Decision

**GREEN** (completed)

## Failure Analysis

(无)

## Do Not Repeat

生产权重/尺度冻结后不得回改 (method/weight/scales/schema 全部冻结)

## 复现入口

- Scripts: `scripts/46_production_table.py scripts/47_cbv2_fusion.py scripts/48_realmlp_prod_local.py scripts/48b_realmlp_cosine_prod.py`
- Reports: `docs/c4-clean-baseline-results.md`
- Artifacts: `output/c4_* output/kaggle_c4_* output/smoke_c4`
