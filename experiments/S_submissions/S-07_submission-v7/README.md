# S-07 — v7 提交 (RealMLP 复刻融合)

## Metadata

- ID: `S-07` (canonical)
- Phase: S_submissions | Created: 2026-08-12
- Status: `completed` | Decision: **GREEN**
- Parent: P1-03 | Successor: P2-01|S-08
- Aliases: SUB-v7
- Tags: submission

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | v5 0.125 |
| Score | blend 0.139683 (PSEUDO) |
| Delta | +0.010 (LB) |
| Public LB | 0.135 |

## Decision

**GREEN** (completed)

## Failure Analysis

(无)

## Do Not Repeat

(无特别禁止项)

## 复现入口

- Scripts: `scripts/40_build_0726.py scripts/41_realmlp_local.py scripts/42_realmlp_fusion.py`
- Reports: `RESULTS.md`
- Artifacts: `output/submissions/submission_blend_v7_rl15/20/25.csv`
