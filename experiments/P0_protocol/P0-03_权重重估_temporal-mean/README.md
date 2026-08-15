# P0-03 — 权重重估 temporal-mean

> 阶段: P0 Protocol 验证 (2026-08-11) | 日期: 2026-08-11 | 状态: **RED**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
temporal fold 均值最优权重?

## Hypothesis
MLP 降权 CatBoost 升权

## Motivation
P0-01

## Data
90 features

## Validation Protocol
Pseudo-LB38

## Method
temporal-mean 权重重估

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | 0.129155 |
| 实验分数 | 0.130015 |
| Delta | +0.0009 |
| Public LB | 0.122 (v3) |

## Decision
**RED**

## Failure Analysis
三次提交 v1/v2/v3 全 0.122 → 表格路线平台期确认, 权重微调无法突破

## Do Not Repeat
不再做表格模型权重微调

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: P1 结构性变化

## 复现入口
- Scripts: `scripts/21_weight_reopt.py scripts/22_final_submission_v3.py`
- Outputs: `output/submissions/submission_blend_v3.csv`
- Reports: `RESULTS.md`
