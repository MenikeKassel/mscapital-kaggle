# C3 — C3 Clean Table v2

> 阶段: C 系列 Clean Baseline v2 (2026-08-13) | 日期: 2026-08-13 | 状态: **GREEN**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
干净表格管线 (LGB+Cat+3seed-MLP) 的四折表现?

## Hypothesis
PSEUDO 0.135051 与 legacy v5 一致

## Motivation
Protocol-v2 表格侧

## Data
90 features

## Validation Protocol
四折可信 outer

## Method
LGBM/CatBoost/3-seed MLP 固定 0.2/0.5/0.3

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | legacy v5 0.134871 |
| 实验分数 | PSEUDO 0.135051 |
| Delta | +0.000180 |
| Public LB | - |

## Decision
**GREEN**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: C4

## 复现入口
- Scripts: `scripts/build_kaggle_c3.py`
- Outputs: `output/c3_* output/kaggle_c3_*`
- Reports: `docs/c3-clean-table*.md`
