# P3-03 — TinyLOBERT 掩码预训练

> 阶段: P3 下一代方法 (2026-08-14) | 日期: 2026-08-14 | 状态: **RED**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
掩码自监督 latent 是否独立?

## Hypothesis
预训练给微调提速

## Motivation
外部方法 (LOBERT)

## Data
LOB 序列

## Validation Protocol
corr 门禁

## Method
Tiny 掩码 transformer

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | - |
| 实验分数 | corr 0.86~0.98 与现有特征 |
| Delta | - |
| Public LB | - |

## Decision
**RED**

## Failure Analysis
掩码 latent 无独立信息 (corr 过高), 预注册门禁终止

## Do Not Repeat
不再做掩码预训练 latent

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: (无直接后继 — 见 experiment-index.md)

## 复现入口
- Scripts: `scripts/p3_03_masked.py`
- Outputs: `output/p3_03_*`
- Reports: `docs/p3-results.md`
