# P3-02 — 状态条件化 (E01 延伸)

> 阶段: P3 下一代方法 (2026-08-14) | 日期: 2026-08-14 | 状态: **RED**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
状态条件化能否释放新信息?

## Hypothesis
四折全正

## Motivation
E01/E02 延续

## Data
152 + 状态特征

## Validation Protocol
PSEUDO/H2/T3/T4

## Method
条件化拼接

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | canonical |
| 实验分数 | PSEUDO +0.00095 |
| Delta | +0.00095 |
| Public LB | - |

## Decision
**RED**

## Failure Analysis
四折全正但拼接稀释 E01 纯状态信号, 未过 gate

## Do Not Repeat
不再用简单拼接做条件化 (FiLM 式融合留给完整版)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: (无直接后继 — 见 experiment-index.md)

## 复现入口
- Scripts: `scripts/p3_02_conditioned.py`
- Outputs: `output/p3_02_*`
- Reports: `docs/p3-results.md`
