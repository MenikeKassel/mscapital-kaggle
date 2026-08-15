# P5-02I — 信息审计 (五条实锤)

> 阶段: P5 市场探针 (2026-08-14/15) | 日期: 2026-08-14 | 状态: **GREEN**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
v7 预测的信息结构: 方向/幅度/分层?

## Hypothesis
幅度富矿 0.466, 方向微弱

## Motivation
P5-01 + GPT1 猜想

## Data
152 features + v7 OOF

## Validation Protocol
FROZEN 51-70 破坏实验

## Method
逐臂破坏信息面 (corr(y)/R/corr(v7)/月度/lo-hi)

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | frozen 0.14849 |
| 实验分数 | |y| corr 0.466 (幅度巨大); sign AUC 0.564 (方向微弱); corr(v7) 高 |
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
- Next: P5-A (幅度门控) / P5-B (条件创新) / P5-C (相位)

## 复现入口
- Scripts: `scripts/p5_02i_info_audit.py`
- Outputs: `output/p5_02i_info_audit`
- Reports: `docs/p5-02i-info-audit-report.md`
