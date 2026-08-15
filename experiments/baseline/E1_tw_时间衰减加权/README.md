# E1_tw — 时间衰减加权

> 阶段: Baseline 表格阶梯 (2026-08-10/11) | 日期: 2026-08-10 | 状态: **RED**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
近期样本加权能否桥接漂移?

## Hypothesis
时间衰减有效

## Motivation
漂移观察

## Data
90 features

## Validation Protocol
CV1

## Method
线性2x/3x, 指数温和/强

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | 0.130204 |
| 实验分数 | 0.125001~0.130161 |
| Delta | - |
| Public LB | - |

## Decision
**RED**

## Failure Analysis
全部无效或有害: 旧月份数据量价值更大, 漂移不是简单时间距离

## Do Not Repeat
不再试任何时间衰减加权

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: P0-2 对抗验证(解释漂移机制)

## 复现入口
- Scripts: `scripts/09_exp_time_weight.py`
- Outputs: `-`
- Reports: `RESULTS.md`
