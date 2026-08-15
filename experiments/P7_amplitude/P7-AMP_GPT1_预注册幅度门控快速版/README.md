# P7-AMP — GPT1 预注册幅度门控快速版

> 阶段: P7 幅度门控终裁 (2026-08-15) | 日期: 2026-08-15 | 状态: **RED**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
gate(mid_range/波动特征) 逐样本加权有无增益?

## Hypothesis
RED 路径 (α=0)

## Motivation
GPT1 round2 评审 + P5-A 终裁

## Data
canonical OOF + 5 amplitude 特征

## Validation Protocol
嵌套 (gate+α 拟合 m21-50, frozen m51-70)

## Method
arm C (无 mid_range) / D (含 mid_range), α 网格

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | arm A 0.14849 |
| 实验分数 | ΔD = +0.00000 (α* 全 0.0) |
| Delta | +0.00000 |
| Public LB | - |

## Decision
**RED**

## Failure Analysis
α 曲线单调下降; 高波动样本方向质量差 (cos 0.215→0.11); baseline 幅度分布已隐式最优

## Do Not Repeat
不再做 volatility confidence calibration / Market→Confidence 假设

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: O→T lag response (GPT P1.5)

## 复现入口
- Scripts: `scripts/p7amp_audit.py scripts/p7amp_quick.py`
- Outputs: `output/p7amp_*.parquet output/p7amp_quick.json`
- Reports: `docs/p7amp-quick-results.md`
