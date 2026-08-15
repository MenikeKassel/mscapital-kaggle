# P4-10 — Loss ablation (MSE vs cosine)

## Metadata

- ID: `P4-10` (canonical)
- Phase: P4_hidden | Created: 2026-08-14
- Status: `superseded` | Decision: **RED**
- Parent: P2-01 | Successor: P4-11|P4-14
- Aliases: P4-08A
- Tags: cosine|objective

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | MSE |
| Score | 本地提升来自验证偏差 |
| Delta | - |
| Public LB | - |

## Decision

**RED** (superseded)

## Failure Analysis

P4-08A 验证偏差: 本地提升来自验证集重用, 非真实增益 (N 系列)

## Do Not Repeat

消融必须严格 nested, 不得重用验证期调参

## 复现入口

- Scripts: `scripts/p4_08a_loss_ablation.py`
- Reports: `docs/p4-hidden-information-report.md §12`
- Artifacts: `output/p4_08a_loss_ablation`
