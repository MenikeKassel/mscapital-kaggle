# H1 — 五模型融合 (CatBoost 惊喜)

## Metadata

- ID: `H1` (legacy)
- Phase: baseline | Created: 2026-08-11
- Status: `completed` | Decision: **RED**
- Parent: G3 | Successor: S-02
- Aliases: -
- Tags: blend

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | 0.138931 |
| Score | 0.139332 |
| Delta | +0.0004 |
| Public LB | 0.122 (v2) |

## Decision

**RED** (completed)

## Failure Analysis

CV +0.0004 未转化为 LB 提升 → 表格特征+树+MLP 组合已饱和 (0.122 平台期)

## Do Not Repeat

不再在表格特征+树+MLP 组合上做小增量融合

## 复现入口

- Scripts: `scripts/17_exp_5model.py scripts/18_final_submission_v2.py`
- Reports: `RESULTS.md`
- Artifacts: `output/submissions/submission_blend_v2.csv`
