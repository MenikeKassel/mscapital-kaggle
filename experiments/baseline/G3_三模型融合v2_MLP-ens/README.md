# G3 — 三模型融合v2 (MLP-ens)

> 阶段: Baseline 表格阶梯 (2026-08-10/11) | 日期: 2026-08-10 | 状态: **GREEN**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
用 MLP-ens 替换单 MLP 的融合?

## Hypothesis
超越社区 CV4 最佳

## Motivation
F2/G2

## Data
90 features

## Validation Protocol
CV1

## Method
LGB+XGB+MLP-ens 0.1/0.4/0.5

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | 0.137710 |
| 实验分数 | 0.138931 |
| Delta | +0.0012 |
| Public LB | 0.122 (v1, #79) |

## Decision
**GREEN**

## Failure Analysis
(无 — 实验通过/非失败)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: v1 提交 → H1

## 复现入口
- Scripts: `scripts/15_exp_g3_blend_ens.py scripts/16_final_submission.py`
- Outputs: `output/submissions/submission_blend_v1.csv`
- Reports: `RESULTS.md`
