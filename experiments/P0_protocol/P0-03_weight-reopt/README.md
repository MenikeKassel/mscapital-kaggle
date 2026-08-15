# P0-03 — 权重重估 temporal-mean

## Metadata

- ID: `P0-03` (canonical)
- Phase: P0_protocol | Created: 2026-08-11
- Status: `completed` | Decision: **RED**
- Parent: P0-01 | Successor: S-03
- Aliases: P0-3
- Tags: validation|blend

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | 0.129155 |
| Score | 0.130015 |
| Delta | +0.0009 |
| Public LB | 0.122 (v3) |

## Decision

**RED** (completed)

## Failure Analysis

三次提交 v1/v2/v3 全 0.122 → 表格路线平台期确认, 权重微调无法突破

## Do Not Repeat

不再做表格模型权重微调

## 复现入口

- Scripts: `scripts/21_weight_reopt.py scripts/22_final_submission_v3.py`
- Reports: `RESULTS.md`
- Artifacts: `output/submissions/submission_blend_v3.csv`
