# G1 — LGBM+MLP 融合

> 阶段: Baseline 表格阶梯 (2026-08-10/11) | 日期: 2026-08-10 | 状态: **GREEN**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
异质模型融合有无大增益?

## Hypothesis
融合是最大杠杆

## Motivation
F1

## Data
90 features

## Validation Protocol
CV1

## Method
w 网格 (corr 0.84)

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | 单 0.1308/0.1321 |
| 实验分数 | 0.137041 |
| Delta | +0.005 |
| Public LB | - |

## Decision
**GREEN**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: F2/G2

## 复现入口
- Scripts: `scripts/11_exp_blend.py`
- Outputs: `-`
- Reports: `RESULTS.md`
