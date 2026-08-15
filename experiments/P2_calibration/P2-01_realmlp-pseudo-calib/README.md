# P2-01 — RealMLP PSEUDO 定标 + 尺度门禁

## Metadata

- ID: `P2-01` (canonical)
- Phase: P2_calibration | Created: 2026-08-12
- Status: `completed` | Decision: **GREEN**
- Parent: S-07 | Successor: C-01|P4-10
- Aliases: P2
- Tags: calibration

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | v5 table PSEUDO 0.134871 |
| Score | RealMLP 0.138560; v7 融合 0.139683 |
| Delta | +0.0048 |
| Public LB | v7 LB 0.135 |

## Decision

**GREEN** (completed)

## Failure Analysis

(无)

## Do Not Repeat

(无特别禁止项)

## 复现入口

- Scripts: `scripts/kaggle_realmlp_pseudo/ (kernel)`
- Reports: `RESULTS.md §P2`
- Artifacts: `output/rlps_v12/`
