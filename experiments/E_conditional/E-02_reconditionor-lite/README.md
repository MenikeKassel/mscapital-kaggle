# E-02 — Reconditionor-lite

## Metadata

- ID: `E-02` (canonical)
- Phase: E_conditional | Created: 2026-08-13
- Status: `completed` | Decision: **RED**
- Parent: E-01 | Successor: -
- Aliases: -
- Tags: conditional|negative

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | canonical |
| Score | 窗口内 cos 0.013, 跨月不可变现 |
| Delta | ~0 |
| Public LB | - |

## Decision

**RED** (completed)

## Failure Analysis

窗口内可解释 ≠ 跨月可预测增量 (负面链 #2)

## Do Not Repeat

(无特别禁止项)

## 复现入口

- Scripts: `scripts/e02_* (src/mscapital/models/context_shift.py)`
- Reports: `docs/e01-e02-e03-results.md`
- Artifacts: `output/e02_*`
