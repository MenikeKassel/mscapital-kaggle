# MSCapital — Real Financial Market Forecasting

这是 Kaggle 社区赛的研究仓库：从高频 market/order/transaction 数据预测未来收益，指标是未中心化 cosine similarity。

## 当前结论（2026-08-21）

- external-assisted 轨：v8b/lb142，Public LB **0.142**。
- self-owned 轨：P6-ORIG，Public LB **0.132**。
- 冻结资产：Clean Baseline v2、152+73Z。
- 当前无运行中的实验、预测或 Kaggle submission。

M01–M05 residual 表示、TCN、无监督 latent、幅度 gate、残差检索、NCL/V-REx 均未通过稳定门禁；Cancel 已证明是 Z 的重叠信息。M06 因缺少 train/test 共有资产/时间键而 `not_identifiable`。E01 ReVol-lite 四折均正但 PSEUDO 未达晋级线，E02/E03 仅为诊断。P10 RQ 生产 run 因跨历史定标不一致而 `protocol_invalid`。

## 推荐下一方向

科学探索优先 BLSM-G1；工程提分第二顺位是 RealMLP recipe 组合；H7 refit ensemble 仅用于降低方差。本阶段不启动实验，预注册见 [`docs/next-direction-2026-08-21.md`](docs/next-direction-2026-08-21.md)。

## SSOT 导航

权威链为：[`CONTEXT.md`](CONTEXT.md) → [`experiments/registry.csv`](experiments/registry.csv) → [`experiments/routes.yaml`](experiments/routes.yaml) → [`submissions/registry.csv`](submissions/registry.csv) → [`docs/project-status.md`](docs/project-status.md) → 生成视图 → 单实验报告 → [`RESULTS.md`](RESULTS.md)（append-only 历史日志）。

- [全部 116 条实验](docs/all-experiments.md)
- [方法路线地图](docs/method-map.md)
- [实验血缘 DAG](docs/experiment-lineage.md)
- [失败与不足结果库](docs/failed-experiments.md)
- [当前研究队列](docs/current-research-queue.md)
- [提交登记](submissions/README.md)

数据、预测、模型权重、Kaggle token 和本地绝对路径不进入公开仓库；它们只作为 registry 中的脱敏 evidence scope 记录。
