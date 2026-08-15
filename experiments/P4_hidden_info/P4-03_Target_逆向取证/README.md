# P4-03 — Target 逆向取证

> 阶段: P4 隐藏信息调查 (2026-08-13/14) | 日期: 2026-08-14 | 状态: **YELLOW**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
target 生成公式 (T1 未来 mid 收益?)

## Hypothesis
y ≈ 未来 mid 收益

## Motivation
H3

## Data
label+market

## Validation Protocol
诊断

## Method
T1-T5 假设检验

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | - |
| 实验分数 | y 与窗口内趋势相关, std∝波动率 |
| Delta | - |
| Public LB | - |

## Decision
**YELLOW**

## Failure Analysis
公式未唯一确定

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: P4-05

## 复现入口
- Scripts: `scripts/p4_03_target_forensics.py`
- Outputs: `output/p4_03_*`
- Reports: `docs/p4-hidden-information-report.md`
