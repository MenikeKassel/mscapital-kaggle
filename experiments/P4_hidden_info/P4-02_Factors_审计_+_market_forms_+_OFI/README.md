# P4-02 — Factors 审计 + market forms + OFI

> 阶段: P4 隐藏信息调查 (2026-08-13/14) | 日期: 2026-08-14 | 状态: **YELLOW**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
未知 factors 表 (H4 第二半) 是否存在痕迹?

## Hypothesis
OFI 形式特征有独立信息

## Motivation
H4

## Data
market/order

## Validation Protocol
multi-outer

## Method
factor audit / market forms / OFI protocol

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | 152 |
| 实验分数 | OFI 弱正 |
| Delta | ~0 |
| Public LB | - |

## Decision
**YELLOW**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: (无直接后继 — 见 experiment-index.md)

## 复现入口
- Scripts: `scripts/p4_02a_factor_audit.py scripts/p4_02b_market_forms.py scripts/p4_02c_ofi_protocol.py`
- Outputs: `output/p4_02*`
- Reports: `docs/p4-hidden-information-report.md`
