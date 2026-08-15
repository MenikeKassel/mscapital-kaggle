# F1 — 轻量 MLP (90特征)

## Metadata

- ID: `F1` (legacy)
- Phase: baseline | Created: 2026-08-10
- Status: `completed` | Decision: **YELLOW**
- Parent: B1 | Successor: F2|G1
- Aliases: -
- Tags: mlp

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | 0.130204 |
| Score | 0.132065 |
| Delta | +0.0019 |
| Public LB | - |

## Decision

**YELLOW** (completed)

## Failure Analysis

loss 未收敛有空间

## Do Not Repeat

(无特别禁止项)

## 复现入口

- Scripts: `scripts/10_exp_mlp.py`
- Reports: `RESULTS.md`
- Artifacts: `-`
