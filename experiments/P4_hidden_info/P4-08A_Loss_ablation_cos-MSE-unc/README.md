# P4-08A — Loss ablation (cos/MSE/unc)

> 阶段: P4 隐藏信息调查 (2026-08-13/14) | 日期: 2026-08-14 | 状态: **INVALID**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
cosine loss vs MSE 在冻结期的真实增益?

## Hypothesis
cosine loss 优于 MSE

## Motivation
社区 (Spiritmilk) + 协议审计

## Data
152 features

## Validation Protocol
受控消融

## Method
loss 消融 + 不确定性消融

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | MSE |
| 实验分数 | 本地提升来自验证偏差 |
| Delta | - |
| Public LB | - |

## Decision
**INVALID**

## Failure Analysis
P4-08A 验证偏差: 本地提升来自验证集重用, 非真实增益 (N 系列)

## Do Not Repeat
消融必须严格 nested, 不得重用验证期调参

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: P4-08C/D/E

## 复现入口
- Scripts: `scripts/p4_08a_loss_ablation.py scripts/p4_08a_unc.py scripts/p4_08a_verify.py scripts/p4_08c_blend.py scripts/p4_08d_simple_cosine_prod.py scripts/p4_08e_v7like_check.py`
- Outputs: `output/p4_08a*`
- Reports: `docs/p4-hidden-information-report.md`
