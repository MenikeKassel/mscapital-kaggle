# B1_ab — 特征组消融 LGO

> 阶段: Baseline 表格阶梯 (2026-08-10/11) | 日期: 2026-08-10 | 状态: **GREEN**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
哪组特征贡献最大?

## Hypothesis
窗口统计是主力

## Motivation
上一个实验结论

## Data
90 features 6组

## Validation Protocol
CV1 逐一移除

## Method
leave-one-group-out

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | 0.130204 |
| 实验分数 | 窗口-0.0102/盘口-0.0042/订单-0.0035/交叉-0.0003/成交-0.0000/EWM+0.0000 |
| Delta | - |
| Public LB | - |

## Decision
**GREEN**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: B2

## 复现入口
- Scripts: `scripts/05_exp_feature_ablation.py`
- Outputs: `-`
- Reports: `RESULTS.md`
