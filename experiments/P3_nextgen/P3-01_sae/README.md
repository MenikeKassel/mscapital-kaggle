# P3-01 — SAE 自编码器表示

## Metadata

- ID: `P3-01` (canonical)
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
| Baseline | canonical |
| Score | PSEUDO -0.00076 |
| Delta | -0.00076 |
| Public LB | - |

## Decision

**RED** (completed)

## Failure Analysis

latent 与 152 手工特征重叠, 无新信息

## Do Not Repeat

不再做无监督 latent 直接进残差学习

## 复现入口

- Scripts: `scripts/p3_01_sae.py scripts/p3_common.py`
- Reports: `docs/p3-results.md`
- Artifacts: `output/p3_01_*`
