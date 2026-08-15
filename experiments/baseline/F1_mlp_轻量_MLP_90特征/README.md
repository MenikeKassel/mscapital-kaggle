# F1_mlp — 轻量 MLP (90特征)

> 阶段: Baseline 表格阶梯 (2026-08-10/11) | 日期: 2026-08-10 | 状态: **YELLOW**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
表格 NN 能否超越 GBDT?

## Hypothesis
MLP 温和超越

## Motivation
模型诊断

## Data
90 features (NaN→fill)

## Validation Protocol
CV1

## Method
MLP[256x2]

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | 0.130204 |
| 实验分数 | 0.132065 |
| Delta | +0.0019 |
| Public LB | - |

## Decision
**YELLOW**

## Failure Analysis
loss 未收敛有空间

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: G1 融合

## 复现入口
- Scripts: `scripts/10_exp_mlp.py`
- Outputs: `-`
- Reports: `RESULTS.md`
