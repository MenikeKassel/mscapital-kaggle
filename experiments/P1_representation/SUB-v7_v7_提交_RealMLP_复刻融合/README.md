# SUB-v7 — v7 提交 (RealMLP 复刻融合)

> 阶段: P1 表示与序列 (2026-08-11/12) | 日期: 2026-08-12 | 状态: **GREEN**
> Alias (历史编号): 提交 v7 (RESULTS.md)
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
152 特征 RealMLP 复刻与表格融合能否突破 0.125?

## Hypothesis
+0.010 单步

## Motivation
公开方案 (0726 特征 + RealMLP)

## Data
f0726 152 features

## Validation Protocol
CV 0.1439@80万切分, n_ens=16

## Method
v5表格 0.8 + RealMLP 0.2

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | v5 0.125 |
| 实验分数 | blend 0.139683 (PSEUDO) |
| Delta | +0.010 (LB) |
| Public LB | 0.135 |

## Decision
**GREEN**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: v8b

## 复现入口
- Scripts: `scripts/40_build_0726.py scripts/41_realmlp_local.py scripts/42_realmlp_fusion.py`
- Outputs: `output/submissions/submission_blend_v7_rl15/20/25.csv`
- Reports: `RESULTS.md`
