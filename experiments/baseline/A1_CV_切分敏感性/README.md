# A1 — CV 切分敏感性

> 阶段: Baseline 表格阶梯 (2026-08-10/11) | 日期: 2026-08-10 | 状态: **GREEN**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
CV 结果对 train/valid 切分有多敏感?

## Hypothesis
越近的验证期 CV 越乐观

## Motivation
上一个实验结论

## Data
90 features

## Validation Protocol
4 切分 CV1-CV4

## Method
同特征同参数×4切分

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | CV1 0.1302 |
| 实验分数 | CV2 0.1338/CV3 0.1352/CV4 0.1418 |
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
- Next: 确立 CV1 为主评估

## 复现入口
- Scripts: `scripts/03_exp_cv_splits.py`
- Outputs: `-`
- Reports: `RESULTS.md`
