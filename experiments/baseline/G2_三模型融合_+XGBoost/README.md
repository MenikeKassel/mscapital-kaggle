# G2 — 三模型融合 +XGBoost

> 阶段: Baseline 表格阶梯 (2026-08-10/11) | 日期: 2026-08-10 | 状态: **YELLOW**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
加入 XGB 是否再增?

## Hypothesis
XGB 互补有限(树间相关高)

## Motivation
G1

## Data
90 features

## Validation Protocol
CV1

## Method
blend 0.1/0.4/0.5

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | 0.137041 |
| 实验分数 | 0.137710 |
| Delta | +0.0007 |
| Public LB | - |

## Decision
**YELLOW**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: G3

## 复现入口
- Scripts: `scripts/13_exp_3model_blend.py`
- Outputs: `-`
- Reports: `RESULTS.md`
