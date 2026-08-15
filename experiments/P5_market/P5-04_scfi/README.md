# P5-04 — SCFI 条件创新 (LGB)

## Metadata

- ID: `P5-04` (canonical)
- Phase: P5_market | Created: 2026-08-14
- Status: `completed` | Decision: **GREEN**
- Parent: P5-02 | Successor: P5-06
- Aliases: P5-B|SCFI
- Tags: conditional|order-flow

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | B1 官方参数 152 |
| Score | C 臂 Δ=+0.0075 (17/20 月正, late +0.0091); blendΔ +0.00094 |
| Delta | +0.0075 |
| Public LB | - |

## Decision

**GREEN** (completed)

## Failure Analysis

NN spot-check 无增益 (learner 依赖: LGB 特异)

## Do Not Repeat

(无特别禁止项)

## 复现入口

- Scripts: `scripts/p5b_build_features.py scripts/p5b_scfi.py scripts/p5b_diagnostics.py scripts/p5b_inf_check.py scripts/p5b_null_check.py`
- Reports: `docs/p5b-scfi-report.md`
- Artifacts: `output/p5b_scfi`
