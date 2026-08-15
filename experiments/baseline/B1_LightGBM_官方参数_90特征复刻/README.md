# B1 — LightGBM 官方参数 90特征复刻

> 阶段: Baseline 表格阶梯 (2026-08-10/11) | 日期: 2026-08-10 | 状态: **GREEN**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
官方基线本地能否复现(LB 0.117)?

## Hypothesis
可复现, 作为后续一切实验的锚点

## Motivation
官方基线

## Data
152→90 features

## Validation Protocol
CV1: m0-50/m51-70

## Method
LightGBM 官方参数, best_iter=700

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | - |
| 实验分数 | 0.130204 |
| Delta | - |
| Public LB | 0.117 (官方) |

## Decision
**GREEN**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: A1/A2/B1/B2

## 复现入口
- Scripts: `scripts/b1_official_baseline.py`
- Outputs: `output/submissions/b1_official_lgb.csv`
- Reports: `RESULTS.md`
