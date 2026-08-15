# P0-02 — Adversarial Validation 对抗验证

## Metadata

- ID: `P0-02` (canonical)
- Phase: P0_protocol | Created: 2026-08-11
- Status: `completed` | Decision: **GREEN**
- Parent: P0-01 | Successor: P0-05
- Aliases: -
- Tags: validation|drift

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | - |
| Score | AUC 0.733/0.777/0.766; m_sp_mean 三组 TOP1; m_book 漂移 45.1% |
| Delta | - |
| Public LB | - |

## Decision

**GREEN** (completed)

## Failure Analysis

(无)

## Do Not Repeat

(无特别禁止项)

## 复现入口

- Scripts: `scripts/20_adversarial_validation.py`
- Reports: `RESULTS.md`
- Artifacts: `-`
