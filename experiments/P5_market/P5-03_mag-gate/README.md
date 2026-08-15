# P5-03 — MAG-Gate 幅度门控

## Metadata

- ID: `P5-03` (canonical)
- Phase: P5_market | Created: 2026-08-14
- Status: `completed` | Decision: **RED**
- Parent: P5-02 | Successor: P7-01
- Aliases: P5-A|MAG-Gate
- Tags: amplitude|negative

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | v7 |
| Score | Δ_outer = -0.000146 |
| Delta | -0.000146 |
| Public LB | - |

## Decision

**RED** (completed)

## Failure Analysis

嵌套后 gate≈常数 (std=0.011), 月度 6/20; 置换对照≈0; 幅度可预测 ≠ 加权有效

## Do Not Repeat

不再做简单 volatility/幅度 gate; 不再细扫 α∈[0,1]

## 复现入口

- Scripts: `scripts/p5a_mag_gate.py`
- Reports: `docs/p5a-mag-gate-report.md`
- Artifacts: `output/p5a_mag_gate`
