# S-06 — v6 提交 (TCN 融合, 灾难)

## Metadata

- ID: `S-06` (canonical)
- Phase: S_submissions | Created: 2026-08-15
- Status: `completed` | Decision: **RED**
- Parent: P1-05 | Successor: -
- Aliases: -
- Tags: submission|negative

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline |  |
| Score |  |
| Delta |  |
| Public LB | - |

## Decision

**RED** (completed)

## Failure Analysis

N005: TCN test 分布外严重退化 (PSEUDO +0.004 → LB 0.082), test corr(tab,tcn)=0.03 致命预警

## Do Not Repeat

序列模型不通过 test 侧 corr 结构验证不得进生产融合

## 复现入口

- Scripts: `scripts/37_final_v6.py`
- Reports: `RESULTS.md (N005)`
- Artifacts: `output/submissions/submission_blend_v6.csv`
