# P4-01a — 600s market 长上下文取证

> 阶段: P4 隐藏信息调查 (2026-08-13/14) | 日期: 2026-08-14 | 状态: **GREEN**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
600 秒盘口快照流是否含独立 target 信息 (H4)?

## Hypothesis
外部模型用了我们未建模的 600s 流

## Motivation
LB142 逆向 + H4 假设

## Data
market 600s

## Validation Protocol
5 块滚动 (canonical)

## Method
market 序列 → 轻量模型, 三数输出

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | 152 features |
| 实验分数 | PSEUDO 正, AUC 增量 +0.019 |
| Delta | +0.019 AUC |
| Public LB | - |

## Decision
**GREEN**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: P5-01/P4-02

## 复现入口
- Scripts: `scripts/p4_01a_*.py`
- Outputs: `output/p4_01a*`
- Reports: `docs/p4-01a-market-forensics-report.md`
