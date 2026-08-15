# P5-B — SCFI 条件创新 (LGB)

> 阶段: P5 市场探针 (2026-08-14/15) | 日期: 2026-08-14 | 状态: **GREEN**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
订单流创新 (O−E[O|M]) 是否含独立 alpha?

## Hypothesis
LGB 强 (ΔC +0.0075)

## Motivation
GPT2 SCFI + 条件创新主线

## Data
73 raw O/T 聚合 + Z 创新特征

## Validation Protocol
双 temporal block (B1/B2), Ridge cross-fit nuisance

## Method
Z = 残差/MAD(train 折外), 五臂 LGB

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | B1 官方参数 152 |
| 实验分数 | C 臂 Δ=+0.0075 (17/20 月正, late +0.0091); blendΔ +0.00094 |
| Delta | +0.0075 |
| Public LB | - |

## Decision
**GREEN**

## Failure Analysis
NN spot-check 无增益 (learner 依赖: LGB 特异)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: P5-D (3seed 生产验证) / P5-E (RealMLP)

## 复现入口
- Scripts: `scripts/p5b_build_features.py scripts/p5b_scfi.py scripts/p5b_diagnostics.py scripts/p5b_inf_check.py scripts/p5b_null_check.py`
- Outputs: `output/p5b_scfi`
- Reports: `docs/p5b-scfi-report.md`
