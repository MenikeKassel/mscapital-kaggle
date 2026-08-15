# P5-07 — RealMLP 生产 learner spot-check

## Metadata

- ID: `P5-07` (canonical)
- Phase: P5_market | Created: 2026-08-15
- Status: `completed` | Decision: **GREEN**
- Parent: P5-06 | Successor: P6-01
- Aliases: P5-E
- Tags: conditional|realmlp

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | A 臂 0.148526 |
| Score | C 0.152570 (Δ+0.004044); blend w=0.50 Δ+0.001369; corr 0.940 |
| Delta | +0.004044 |
| Public LB | - |

## Decision

**GREEN** (completed)

## Failure Analysis

SmallMLP 代理被推翻 (过弱)

## Do Not Repeat

(无特别禁止项)

## 复现入口

- Scripts: `scripts/p5e_prep_realmlp.py scripts/p5e_realmlp_spotcheck.py scripts/p5e_blend_check.py`
- Reports: `docs/p5de-production-verification.md`
- Artifacts: `output/p5e_realmlp_spotcheck`
