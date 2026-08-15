# P3-02 — 状态条件化 (E01 延伸)

## Metadata

- ID: `P3-02` (canonical)
- Phase: P3_nextgen | Created: 2026-08-14
- Status: `completed` | Decision: **RED**
- Parent: C-04 | Successor: -
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
| Score | PSEUDO +0.00095 |
| Delta | +0.00095 |
| Public LB | - |

## Decision

**RED** (completed)

## Failure Analysis

四折全正但拼接稀释 E01 纯状态信号, 未过 gate

## Do Not Repeat

不再用简单拼接做条件化 (FiLM 式融合留给完整版)

## 复现入口

- Scripts: `scripts/p3_02_conditioned.py`
- Reports: `docs/p3-results.md`
- Artifacts: `output/p3_02_*`
