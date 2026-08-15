# P5-C — RICS 跨通道几何

> 阶段: P5 市场探针 (2026-08-14/15) | 日期: 2026-08-14 | 状态: **RED**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
最后 10 步跨通道形态 (lag-cov/相干) 有无 alpha?

## Hypothesis
有 (GPT2 RICS)

## Motivation
GPT2 RICS + REVCOH

## Data
market last-10 步 (30s)

## Validation Protocol
五层阶梯, frozen 51-70

## Method
flatten/moments/cov/lag/phase 阶梯

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | market 基线 0.086 |
| 实验分数 | 全 ≤0.011 corr_y; R4 相位 -0.006 (反转) |
| Delta | - |
| Public LB | - |

## Decision
**RED**

## Failure Analysis
短窗形态无信息; 相位破坏反演不变; M0-ref 复现 0.0861 确认协议有效

## Do Not Repeat
不再做 wavelet/shapelet/spectral CNN 短窗形态

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: (无直接后继 — 见 experiment-index.md)

## 复现入口
- Scripts: `scripts/p5c_rics.py`
- Outputs: `output/p5c_rics`
- Reports: `docs/p5c-rics-report.md`
