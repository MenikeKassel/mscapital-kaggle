# P0-04 — MLP Fairness Check (last-epoch bug)

## Metadata

- ID: `P0-04` (canonical)
- Phase: P0_protocol | Created: 2026-08-11
- Status: `superseded` | Decision: **GREEN**
- Parent: P0-01 | Successor: -
- Aliases: P0.5-B
- Tags: protocol|mlp

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | T1 0.11062 |
| Score | T1 +0.0062 / T2 +0.0045 / T3 +0.0061 |
| Delta | +0.0050 平均 |
| Public LB | - |

## Decision

**GREEN** (superseded)

## Failure Analysis

评估协议高估其劣势; 修正后 MLP 仍弱于 CatBoost (T2 差 0.0054)

## Do Not Repeat

评估模型必须用 best-state 而非 last-epoch

## 复现入口

- Scripts: `scripts/24_mlp_fairness.py`
- Reports: `RESULTS.md`
- Artifacts: `-`
