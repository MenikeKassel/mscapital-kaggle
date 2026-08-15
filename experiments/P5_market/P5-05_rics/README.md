# P5-05 — RICS 跨通道几何

## Metadata

- ID: `P5-05` (canonical)
- Phase: P5_market | Created: 2026-08-14
- Status: `completed` | Decision: **RED**
- Parent: P5-02 | Successor: -
- Aliases: P5-C|RICS
- Tags: geometry|negative

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | market 基线 0.086 |
| Score | 全 ≤0.011 corr_y; R4 相位 -0.006 (反转) |
| Delta | - |
| Public LB | - |

## Decision

**RED** (completed)

## Failure Analysis

短窗形态无信息; 相位破坏反演不变; M0-ref 复现 0.0861 确认协议有效

## Do Not Repeat

不再做 wavelet/shapelet/spectral CNN 短窗形态

## 复现入口

- Scripts: `scripts/p5c_rics.py`
- Reports: `docs/p5c-rics-report.md`
- Artifacts: `output/p5c_rics`
