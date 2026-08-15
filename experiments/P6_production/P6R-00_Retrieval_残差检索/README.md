# P6R-00 — Retrieval 残差检索

> 阶段: P6/P6R 生产与检索 (2026-08-14/15) | 日期: 2026-08-14 | 状态: **RED**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
相似市场状态的残差均值可否跨月迁移?

## Hypothesis
检索残差有预测增量

## Motivation
E02 窗口内 cos 0.013 + M05

## Data
E02 11 context 特征 + canonical OOF

## Validation Protocol
8 候选 (K×度量) × 4 折, 严格时序 KNN, bootstrap CI 门禁

## Method
FAISS 时序 KNN → 残差均值 r̂, y = RMS(y0)+α·RMS(r̂)

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | RMS frozen |
| 实验分数 | 32/32 候选全正, 最大 PSEUDO Δ=+0.000588 (gate 39%) |
| Delta | +0.000588 |
| Public LB | - |

## Decision
**RED**

## Failure Analysis
无候选同时满足 CI 下界>0 与 α≥0.10; 相似状态残差均值预测力弱; 负面链 #5

## Do Not Repeat
不再做检索-残差预测路线

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: P6R-01 终裁 (挂起, 等拍板)

## 复现入口
- Scripts: `scripts/p6r_00_retrieval_residual.py scripts/p6r_00_variants.py`
- Outputs: `output/p6r_00`
- Reports: `docs/p6r_experiment_report.md`
