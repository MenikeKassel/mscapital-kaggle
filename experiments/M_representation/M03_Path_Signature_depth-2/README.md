# M03 — Path Signature (depth-2)

> 阶段: M 系列残差表示 (2026-08-13) | 日期: 2026-08-13 | 状态: **RED**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
路径签名能否捕捉价格路径结构?

## Hypothesis
签名特征有残差 alpha

## Motivation
表示探索

## Data
market 路径

## Validation Protocol
canonical residual OOF

## Method
depth-2 path signature → 残差学习

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | canonical |
| 实验分数 | ~0 |
| Delta | ~0 |
| Public LB | - |

## Decision
**RED**

## Failure Analysis
签名特征无残差增量

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: (无直接后继 — 见 experiment-index.md)

## 复现入口
- Scripts: `scripts/m03_* (src/mscapital/features/path_signature.py)`
- Outputs: `output/m03_*`
- Reports: `docs/m03-path-signature-results.md`
