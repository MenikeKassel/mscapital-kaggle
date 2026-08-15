# P1-05 — TCN 双塔序列模型

## Metadata

- ID: `P1-05` (canonical)
- Phase: P1_representation | Created: 2026-08-11
- Status: `completed` | Decision: **RED**
- Parent: P1-03 | Successor: S-06
- Aliases: -
- Tags: sequence|negative

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | 表格 blend |
| Score | PSEUDO 0.0738 单模; 3-fold 融合全正 |
| Delta | +0.0035/+0.0023/+0.0044 |
| Public LB | 0.082 (v6, 灾难) |

## Decision

**RED** (completed)

## Failure Analysis

N005: test 分布外严重退化 — PSEUDO 验证对 TCN 不可靠 (test corr(tab,tcn)=0.03 是致命预警); 序列模型完全不可信

## Do Not Repeat

不再用序列模型做生产融合 (除非 test 侧 corr 结构验证)

## 复现入口

- Scripts: `scripts/30_p12_local_run.py scripts/32_blend_tcn.py scripts/33_p12_enhance.py scripts/34_fusion_3fold.py scripts/36_tcn_full.py scripts/37_final_v6.py`
- Reports: `RESULTS.md (N005)`
- Artifacts: `output/p12* output/submissions/submission_blend_v6.csv`
