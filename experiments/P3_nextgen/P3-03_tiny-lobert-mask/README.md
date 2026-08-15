# P3-03 — TinyLOBERT 掩码预训练

## Metadata

- ID: `P3-03` (canonical)
- Phase: P3_nextgen | Created: 2026-08-14
- Status: `completed` | Decision: **RED**
- Parent: C-04 | Successor: -
- Aliases: -
- Tags: unsupervised|negative

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | - |
| Score | corr 0.86~0.98 与现有特征 |
| Delta | - |
| Public LB | - |

## Decision

**RED** (completed)

## Failure Analysis

掩码 latent 无独立信息 (corr 过高), 预注册门禁终止

## Do Not Repeat

不再做掩码预训练 latent

## 复现入口

- Scripts: `scripts/p3_03_masked.py`
- Reports: `docs/p3-results.md`
- Artifacts: `output/p3_03_*`
