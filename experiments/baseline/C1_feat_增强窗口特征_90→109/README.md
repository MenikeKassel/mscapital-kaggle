# C1_feat — 增强窗口特征 90→109

> 阶段: Baseline 表格阶梯 (2026-08-10/11) | 日期: 2026-08-10 | 状态: **RED**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
更多窗口统计特征还有增益吗?

## Hypothesis
特征工程到顶

## Motivation
EDA

## Data
109 features

## Validation Protocol
CV1

## Method
+窗口5/30/300+偏度/分位/比率

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | 0.130204 |
| 实验分数 | 0.129760 |
| Delta | -0.0004 |
| Public LB | - |

## Decision
**RED**

## Failure Analysis
特征工程到顶, 90 特征即甜点位

## Do Not Repeat
不再盲目堆窗口统计特征

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: 转向模型侧 (D1)

## 复现入口
- Scripts: `scripts/07_exp_enhanced_windows.py`
- Outputs: `-`
- Reports: `RESULTS.md`
