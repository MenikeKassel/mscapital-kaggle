# V8 — v8/v8b 提交 (lb142 推理包融合)

> 阶段: P1 表示与序列 (2026-08-11/12) | 日期: 2026-08-12 | 状态: **GREEN**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
外部开源 LB142 推理包融合能否追平最强公开方案?

## Hypothesis
0.5/0.5 甜点位

## Motivation
外部方案 (yangq369 submit-lb142)

## Data
外部预测 (MultiStream+RealMLP v9/v10)

## Validation Protocol
Public LB

## Method
v7 + lb142ref 0.5/0.5 (v8a 0.7/0.3=0.139)

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | v7 0.135 |
| 实验分数 | - |
| Delta | +0.007 |
| Public LB | 0.142 (#30) |

## Decision
**GREEN**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: P2 校准接手

## 复现入口
- Scripts: `scripts/43_lb142_fusion.py scripts/44_threeway_fusion.py`
- Outputs: `output/submissions/submission_v8_ref*.csv output/submissions/submission_v9_*.csv`
- Reports: `RESULTS.md`
