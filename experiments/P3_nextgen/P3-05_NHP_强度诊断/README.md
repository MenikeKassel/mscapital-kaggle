# P3-05 — NHP 强度诊断

> 阶段: P3 下一代方法 (2026-08-14) | 日期: 2026-08-14 | 状态: **RED**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
Hawkes 过程强度对 target 有无预测力?

## Hypothesis
|r| ≤ 0.01 极弱

## Motivation
点过程理论 (sd-PNHP)

## Data
order/tx 事件流

## Validation Protocol
诊断

## Method
NHP 强度估计 vs target 相关

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | - |
| 实验分数 | |r|≤0.01 |
| Delta | - |
| Public LB | - |

## Decision
**RED**

## Failure Analysis
诊断通过但信号极弱, 不投入完整 NHP

## Do Not Repeat
不投入完整 NHP 模型

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: (无直接后继 — 见 experiment-index.md)

## 复现入口
- Scripts: `scripts/p3_05_nhp_diag.py`
- Outputs: `output/p3_05_*`
- Reports: `docs/p3-results.md`
