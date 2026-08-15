# E1-TW — 时间衰减加权

## Metadata

- ID: `E1-TW` (legacy)
- Phase: baseline | Created: 2026-08-10
- Status: `completed` | Decision: **RED**
- Parent: B1 | Successor: -
- Aliases: -
- Tags: validation

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | 0.130204 |
| Score | 0.125001~0.130161 |
| Delta | - |
| Public LB | - |

## Decision

**RED** (completed)

## Failure Analysis

全部无效或有害: 旧月份数据量价值更大, 漂移不是简单时间距离

## Do Not Repeat

不再试任何时间衰减加权

## 复现入口

- Scripts: `scripts/09_exp_time_weight.py`
- Reports: `RESULTS.md`
- Artifacts: `-`
