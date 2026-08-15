# A2 — 特征数量 17 vs 90

> 阶段: Baseline 表格阶梯 (2026-08-10/11) | 日期: 2026-08-10 | 状态: **GREEN**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
特征数量是否是预测力的主成分?

## Hypothesis
增量特征贡献绝大部分

## Motivation
EDA

## Data
90 features

## Validation Protocol
CV1

## Method
17 基础 vs 90 完整

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | 0.0969 |
| 实验分数 | 0.1302 |
| Delta | +0.0333 |
| Public LB | - |

## Decision
**GREEN**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: B1 消融

## 复现入口
- Scripts: `scripts/04_exp_feat_count.py`
- Outputs: `-`
- Reports: `RESULTS.md`
