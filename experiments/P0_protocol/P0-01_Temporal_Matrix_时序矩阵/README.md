# P0-01 — Temporal Matrix 时序矩阵

> 阶段: P0 Protocol 验证 (2026-08-11) | 日期: 2026-08-11 | 状态: **GREEN**
> Alias (历史编号): P0-1 (历史简写)
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
CV1 模型选择在时序 fold 上是否可信?

## Hypothesis
CV1 会翻转 (MLP 假冠军)

## Motivation
v2 提交未提升

## Data
90 features

## Validation Protocol
T1-T4/H1/H2/PSEUDO38 七折

## Method
4 模型 × 7 时序折全矩阵

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | - |
| 实验分数 | MLP 7/7 垫底(0.1187), CatBoost 最稳健(0.1264), blend 每折全胜+0.003 |
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
- Next: P0-03 权重重估

## 复现入口
- Scripts: `scripts/19_temporal_matrix.py`
- Outputs: `-`
- Reports: `RESULTS.md`
