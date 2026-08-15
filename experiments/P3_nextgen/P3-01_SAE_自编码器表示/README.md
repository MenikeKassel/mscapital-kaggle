# P3-01 — SAE 自编码器表示

> 阶段: P3 下一代方法 (2026-08-14) | 日期: 2026-08-14 | 状态: **RED**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
SAE latent 是否含 152 手工特征外的信息?

## Hypothesis
latent 有独立信息

## Motivation
外部方法 (GPT 方案)

## Data
152 features → SAE latent

## Validation Protocol
PSEUDO/H2/T3/T4 四折

## Method
SAE 稀疏编码 → 残差学习

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | canonical |
| 实验分数 | PSEUDO -0.00076 |
| Delta | -0.00076 |
| Public LB | - |

## Decision
**RED**

## Failure Analysis
latent 与 152 手工特征重叠, 无新信息

## Do Not Repeat
不再做无监督 latent 直接进残差学习

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: (无直接后继 — 见 experiment-index.md)

## 复现入口
- Scripts: `scripts/p3_01_sae.py scripts/p3_common.py`
- Outputs: `output/p3_01_*`
- Reports: `docs/p3-results.md`
