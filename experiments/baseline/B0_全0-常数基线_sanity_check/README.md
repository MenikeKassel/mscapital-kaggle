# B0 — 全0/常数基线 sanity check

> 阶段: Baseline 表格阶梯 (2026-08-10/11) | 日期: 2026-08-10 | 状态: **SUPERSEDED**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
预测全 0 与全 -1 时 cosine 分数是多少(社区数据 sanity)?

## Hypothesis
全0=0, 全-1≈-0.007

## Motivation
EDA

## Data
label

## Validation Protocol
公开测试集 LB

## Method
常数预测

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | - |
| 实验分数 | 0 / -0.007 |
| Delta | - |
| Public LB | 0 / -0.007 |

## Decision
**SUPERSEDED**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: B1

## 复现入口
- Scripts: `(见阶段报告)`
- Outputs: `-`
- Reports: `RESULTS.md`
