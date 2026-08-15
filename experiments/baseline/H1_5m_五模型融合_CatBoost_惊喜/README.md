# H1_5m — 五模型融合 (CatBoost 惊喜)

> 阶段: Baseline 表格阶梯 (2026-08-10/11) | 日期: 2026-08-11 | 状态: **RED**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
CatBoost 加入融合?

## Hypothesis
CatBoost 单模型更强

## Motivation
G3

## Data
90 features

## Validation Protocol
CV1

## Method
XGB+Cat+MLP 0.1/0.4/0.5

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | 0.138931 |
| 实验分数 | 0.139332 |
| Delta | +0.0004 |
| Public LB | 0.122 (v2) |

## Decision
**RED**

## Failure Analysis
CV +0.0004 未转化为 LB 提升 → 表格特征+树+MLP 组合已饱和 (0.122 平台期)

## Do Not Repeat
不再在表格特征+树+MLP 组合上做小增量融合

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: P0 协议验证 → 序列模型

## 复现入口
- Scripts: `scripts/17_exp_5model.py scripts/18_final_submission_v2.py`
- Outputs: `output/submissions/submission_blend_v2.csv`
- Reports: `RESULTS.md`
