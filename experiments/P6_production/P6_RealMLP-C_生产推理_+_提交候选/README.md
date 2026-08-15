# P6 — RealMLP-C 生产推理 + 提交候选

> 阶段: P6/P6R 生产与检索 (2026-08-14/15) | 日期: 2026-08-15 | 状态: **YELLOW**
> Alias (历史编号): -
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
SCFI 升级到全量生产 (test 预测) 是否过门禁?

## Hypothesis
双窗口 blendΔ +0.0014 一致 → 提交候选

## Motivation
P5-E 裁决执行

## Data
152+Z features, test 647,896 行

## Validation Protocol
PSEUDO (eval 33-70) + R61_70 双窗口; w 51-60 调

## Method
RealMLP(152+Z) 全量 refit 0-70 → test; blend v8b+0.55×C

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | v8b (PSEUDO canon 0.142550) |
| 实验分数 | PSEUDO blendΔ +0.001435 (w=0.75); R61_70 +0.001369; std 比 0.9726 |
| Delta | +0.0014 |
| Public LB | - |

## Decision
**YELLOW**

## Failure Analysis
未提交 (等用户拍板); 期望 LB 0.1423~0.1428 (v8b 已含 RealMLP 族, 增量压缩)

## Do Not Repeat
(无特别禁止项)

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: 提交拍板 → O→T lag response

## 复现入口
- Scripts: `scripts/p6_build_test_features.py scripts/p6_prod_realmlp.py scripts/p6_finish_test.py scripts/p6_prod_audit.py scripts/p6_prod_audit2.py`
- Outputs: `output/p6_prod/ (submission_candidate_p6.csv)`
- Reports: `docs/p6-production-inference.md`
