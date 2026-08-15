# P7-01 — GPT1 预注册幅度门控快速版

## Metadata

- ID: `P7-01` (canonical)
- Phase: P7_amplitude | Created: 2026-08-15
- Status: `completed` | Decision: **RED**
- Parent: P5-03 | Successor: -
- Aliases: P7-AMP
- Tags: amplitude|negative

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | arm A 0.14849 |
| Score | ΔD = +0.00000 (α* 全 0.0) |
| Delta | +0.00000 |
| Public LB | - |

## Decision

**RED** (completed)

## Failure Analysis

α 曲线单调下降; 高波动样本方向质量差 (cos 0.215→0.11); baseline 幅度分布已隐式最优

## Do Not Repeat

不再做 volatility confidence calibration / Market→Confidence 假设

## 复现入口

- Scripts: `scripts/p7amp_audit.py scripts/p7amp_quick.py`
- Reports: `docs/p7amp-quick-results.md`
- Artifacts: `output/p7amp_*.parquet output/p7amp_quick.json`
