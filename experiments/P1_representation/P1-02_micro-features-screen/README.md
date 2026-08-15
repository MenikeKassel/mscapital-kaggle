# P1-02 — 双轴筛选 (alpha+drift)

## Metadata

- ID: `P1-02` (canonical)
- Phase: P1_representation | Created: 2026-08-11
- Status: `completed` | Decision: **YELLOW**
- Parent: P1-01 | Successor: P1-03
- Aliases: P1-1b|P1-01b
- Tags: features|validation

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | 152 |
| Score | alpha mean +0.0022 |
| Delta | +0.0022 |
| Public LB | - |

## Decision

**YELLOW** (completed)

## Failure Analysis

drift 轴 ΔAUC +0.0168 ⚠️ 需第二轮归一化; 净效果为正

## Do Not Repeat

(无特别禁止项)

## 复现入口

- Scripts: `scripts/28_feature_screening.py`
- Reports: `RESULTS.md`
- Artifacts: `-`
