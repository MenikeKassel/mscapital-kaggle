# G3 — 三模型融合v2 (MLP-ens)

## Metadata

- ID: `G3` (legacy)
- Phase: baseline | Created: 2026-08-10
- Status: `completed` | Decision: **GREEN**
- Parent: G2 | Successor: H1|S-01
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
| Baseline | 0.137710 |
| Score | 0.138931 |
| Delta | +0.0012 |
| Public LB | 0.122 (v1, #79) |

## Decision

**GREEN** (completed)

## Failure Analysis

(无)

## Do Not Repeat

(无特别禁止项)

## 复现入口

- Scripts: `scripts/15_exp_g3_blend_ens.py scripts/16_final_submission.py`
- Reports: `RESULTS.md`
- Artifacts: `output/submissions/submission_blend_v1.csv`
