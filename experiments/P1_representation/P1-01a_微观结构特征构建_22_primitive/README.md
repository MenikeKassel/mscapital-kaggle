# P1-01a — 微观结构特征构建 (22 primitive)

> 阶段: P1 表示与序列 (2026-08-11/12) | 日期: 2026-08-11 | 状态: **GREEN**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
事件流微观结构特征能否提供新信息?

## Hypothesis
22 个无量纲 primitive 有 alpha

## Motivation
P0 平台期 → 结构性变化

## Data
order+transaction

## Validation Protocol
train/test 各 119s 构建

## Method
order: add/cancel imb/OFI norm/arrival/burstiness/大单; tx: 强度/大单/fast-slow; market: microprice/相对价差/L2 深度不平衡

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | - |
| 实验分数 | - |
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
- Next: P1-01b

## 复现入口
- Scripts: `scripts/27_build_micro_features.py`
- Outputs: `output/micro_features_*.parquet (D:/mscapital-forecasting/data/processed)`
- Reports: `RESULTS.md`
