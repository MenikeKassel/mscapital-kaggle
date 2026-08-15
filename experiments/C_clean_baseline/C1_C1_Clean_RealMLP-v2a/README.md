# C1 — C1 Clean RealMLP-v2a

> 阶段: C 系列 Clean Baseline v2 (2026-08-13) | 日期: 2026-08-13 | 状态: **GREEN**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
Protocol-v2 干净复刻的 RealMLP 基准?

## Hypothesis
复现 community RealMLP 水准

## Motivation
Protocol-v2 建立

## Data
152 features

## Validation Protocol
四折可信 outer (PSEUDO/H2/T3/T4)

## Method
RealMLP-v2a (Kaggle P100 双轨)

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | - |
| 实验分数 | ~0.143 (R61_70 复现 canonical 水准) |
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
- Next: C2

## 复现入口
- Scripts: `scripts/build_kaggle_c1.py`
- Outputs: `output/c1_* output/kaggle_c1_*`
- Reports: `docs/c1-clean-realmlp*.md`
