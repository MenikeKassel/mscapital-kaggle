# P0-02 — Adversarial Validation 对抗验证

> 阶段: P0 Protocol 验证 (2026-08-11) | 日期: 2026-08-11 | 状态: **GREEN**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
train/test 漂移的来源与主力特征?

## Hypothesis
漂移集中在价差/深度/波动特征

## Motivation
CV-LB 差 0.017

## Data
90 features

## Validation Protocol
3 组对抗 (A/B/C)

## Method
二分类 AUC + 特征重要性

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | - |
| 实验分数 | AUC 0.733/0.777/0.766; m_sp_mean 三组 TOP1; m_book 漂移 45.1% |
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
- Next: P0.5-C 归一化干预

## 复现入口
- Scripts: `scripts/20_adversarial_validation.py`
- Outputs: `-`
- Reports: `RESULTS.md`
