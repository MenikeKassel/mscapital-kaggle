# P5-A — MAG-Gate 幅度门控

> 阶段: P5 市场探针 (2026-08-14/15) | 日期: 2026-08-14 | 状态: **RED**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
按预测幅度逐样本加权能否提升 cosine?

## Hypothesis
尾部加权有增益 (COC)

## Motivation
P5-02I 幅度富矿 + 四方外部方案

## Data
v7 OOF + m̂

## Validation Protocol
4-bin 嵌套 (gate 只在前月 OOF 拟合), frozen 51-70

## Method
ŷ = w_bin(m̂)·v7, 4-bin + γ 幂族

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | v7 |
| 实验分数 | Δ_outer = -0.000146 |
| Delta | -0.000146 |
| Public LB | - |

## Decision
**RED**

## Failure Analysis
嵌套后 gate≈常数 (std=0.011), 月度 6/20; 置换对照≈0; 幅度可预测 ≠ 加权有效

## Do Not Repeat
不再做简单 volatility/幅度 gate; 不再细扫 α∈[0,1]

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: P7-AMP (GPT 预注册终裁)

## 复现入口
- Scripts: `scripts/p5a_mag_gate.py`
- Outputs: `output/p5a_mag_gate`
- Reports: `docs/p5a-mag-gate-report.md`
