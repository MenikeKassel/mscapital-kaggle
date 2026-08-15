# E01 — ReVol-lite 状态条件化

> 阶段: E 系列状态条件化 (2026-08-13) | 日期: 2026-08-13 | 状态: **YELLOW**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
波动状态条件化能否改善残差预测?

## Hypothesis
+0.0011 但未过 gate

## Motivation
状态条件化首试

## Data
market 状态

## Validation Protocol
multi-outer

## Method
revol_lite 状态表示 → 残差学习

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | canonical |
| 实验分数 | +0.0011 |
| Delta | +0.0011 |
| Public LB | - |

## Decision
**YELLOW**

## Failure Analysis
gate 未过 (上界参考)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: E02 (P3-02 延伸)

## 复现入口
- Scripts: `scripts/e01_* (src/mscapital/models/revol_lite.py)`
- Outputs: `output/e01_*`
- Reports: `docs/e01-e02-e03-results.md`
