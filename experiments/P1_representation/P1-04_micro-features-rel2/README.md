# P1-04 — 特征相对化第二轮

## Metadata

- ID: `P1-04` (canonical)
- Phase: P1_representation | Created: 2026-08-11
- Status: `completed` | Decision: **RED**
- Parent: P1-03 | Successor: -
- Aliases: P1-1e|P1-01e
- Tags: features|negative

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | P1-01c |
| Score | T4 +0.0003 / PSEUDO -0.0012 |
| Delta | -0.0012 (PSEUDO) |
| Public LB | - |

## Decision

**RED** (completed)

## Failure Analysis

相对化在 PSEUDO 反而更差; 第一轮微观特征已是最优

## Do Not Repeat

不再对微观特征做第二轮相对化

## 复现入口

- Scripts: `scripts/38_micro_rel2.py`
- Reports: `RESULTS.md (N006)`
- Artifacts: `-`
