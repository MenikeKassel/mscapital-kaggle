# C4 — C4 Clean Baseline v2 冻结

> 阶段: C 系列 Clean Baseline v2 (2026-08-13) | 日期: 2026-08-13 | 状态: **GREEN**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
生产规则 0.63×RMS(RealMLP)+0.37×RMS(Table) 的四折增益?

## Hypothesis
全折全正, 描述性均值 +0.0050

## Motivation
C2+C3 合并

## Data
152+90 features

## Validation Protocol
inner-only raw/std/RMS 校准, fold-adaptive outer

## Method
RMS 尺度校准融合, fold-adaptive + 固定规则

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | RealMLP 单模 |
| 实验分数 | 0.142649/0.141762/0.143515/0.156924 |
| Delta | +0.005026 mean |
| Public LB | - |

## Decision
**GREEN**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
生产权重/尺度冻结后不得回改 (method/weight/scales/schema 全部冻结)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: P5 探针 (在冻结基线上)

## 复现入口
- Scripts: `scripts/46_production_table.py scripts/47_cbv2_fusion.py scripts/48_realmlp_prod_local.py scripts/48b_realmlp_cosine_prod.py`
- Outputs: `output/c4_* output/kaggle_c4_* output/smoke_c4`
- Reports: `docs/c4-clean-baseline-results.md`
