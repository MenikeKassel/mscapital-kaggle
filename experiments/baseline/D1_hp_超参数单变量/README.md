# D1_hp — 超参数单变量

> 阶段: Baseline 表格阶梯 (2026-08-10/11) | 日期: 2026-08-10 | 状态: **YELLOW**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
官方超参附近单变量调整有无增益?

## Hypothesis
leaves64 小幅提升

## Motivation
模型诊断

## Data
90 features

## Validation Protocol
CV1

## Method
leaves64/128, lr05, L2_20, minleaf100

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | 0.130204 |
| 实验分数 | 0.130763 |
| Delta | +0.0006 (leaves64) |
| Public LB | - |

## Decision
**YELLOW**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: E1

## 复现入口
- Scripts: `scripts/08_exp_hp_sweep.py`
- Outputs: `-`
- Reports: `RESULTS.md`
