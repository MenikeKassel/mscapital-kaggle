# F2 — MLP 30ep×3seed 集成

> 阶段: Baseline 表格阶梯 (2026-08-10/11) | 日期: 2026-08-10 | 状态: **GREEN**
> Alias (历史编号): F2 MLP 集成
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
MLP 多 seed 集成能否拉回弱 seed?

## Hypothesis
集成更稳更强

## Motivation
G1

## Data
90 features

## Validation Protocol
CV1

## Method
3-seed 平均

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | 0.1321/0.1298/0.1287 |
| 实验分数 | 0.133736 |
| Delta | +0.0017 |
| Public LB | - |

## Decision
**GREEN**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: G3

## 复现入口
- Scripts: `scripts/14_exp_mlp_ens.py`
- Outputs: `-`
- Reports: `RESULTS.md`
