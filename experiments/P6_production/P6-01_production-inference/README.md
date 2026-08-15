# P6-01 — RealMLP-C 生产推理 + 提交候选

## Metadata

- ID: `P6-01` (canonical)
- Phase: P6_production | Created: 2026-08-15
- Status: `completed` | Decision: **YELLOW**
- Parent: P5-07 | Successor: -
- Aliases: -
- Tags: production

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | v8b (PSEUDO canon 0.142550) |
| Score | PSEUDO blendΔ +0.001435 (w=0.75); R61_70 +0.001369; std 比 0.9726 |
| Delta | +0.0014 |
| Public LB | - |

## Decision

**YELLOW** (completed)

## Failure Analysis

未提交 (等用户拍板); 期望 LB 0.1423~0.1428 (v8b 已含 RealMLP 族, 增量压缩)

## Do Not Repeat

(无特别禁止项)

## 复现入口

- Scripts: `scripts/p6_build_test_features.py scripts/p6_prod_realmlp.py scripts/p6_finish_test.py scripts/p6_prod_audit.py scripts/p6_prod_audit2.py`
- Reports: `docs/p6-production-inference.md`
- Artifacts: `output/p6_prod/ (submission_candidate_p6.csv)`
