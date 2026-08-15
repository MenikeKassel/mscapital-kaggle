# P1-02 — TCN 双塔序列模型

> 阶段: P1 表示与序列 (2026-08-11/12) | 日期: 2026-08-11 | 状态: **RED**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
事件序列建模能否补充表格?

## Hypothesis
TCN 捕捉表格未覆盖的时序结构

## Motivation
P0 平台期 + 公开方案

## Data
order(60s×16ch)+transaction(60bar×8ch)

## Validation Protocol
PSEUDO/T3/T4

## Method
双塔 TCN 60ep×3seed

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | 表格 blend |
| 实验分数 | PSEUDO 0.0738 单模; 3-fold 融合全正 |
| Delta | +0.0035/+0.0023/+0.0044 |
| Public LB | 0.082 (v6, 灾难) |

## Decision
**RED**

## Failure Analysis
N005: test 分布外严重退化 — PSEUDO 验证对 TCN 不可靠 (test corr(tab,tcn)=0.03 是致命预警); 序列模型完全不可信

## Do Not Repeat
不再用序列模型做生产融合 (除非 test 侧 corr 结构验证)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: v7 RealMLP 复刻 (表格+NN)

## 复现入口
- Scripts: `scripts/30_p12_local_run.py scripts/32_blend_tcn.py scripts/33_p12_enhance.py scripts/34_fusion_3fold.py scripts/36_tcn_full.py scripts/37_final_v6.py`
- Outputs: `output/p12* output/submissions/submission_blend_v6.csv`
- Reports: `RESULTS.md (N005)`
