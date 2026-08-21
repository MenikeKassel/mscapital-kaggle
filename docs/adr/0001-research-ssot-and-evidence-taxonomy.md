# ADR 0001：研究 SSOT 与证据分类

- 状态：Accepted
- 日期：2026-08-21

## 决策

实验事实集中在 `experiments/registry.csv` schema v3；路线状态集中在 `experiments/routes.yaml`；Kaggle 提交集中在 `submissions/registry.csv`。Markdown 视图由生成器重建，`RESULTS.md` 只追加历史记录，不再承担当前状态。

每条记录必须有唯一数字型 canonical ID（旧名称进入 alias），并显式记录 `record_kind`、`evidence_state`、`ownership`、`route_id`、`failure_category`、`provenance_state` 和 `source_refs`。诊断、审计、无效和不可识别记录不得进入候选或提交视图。

## 后果

这套分类允许把“有正相关但不足以晋级”“仅解释机制”“外部 LB 取证”和“协议无效”分开，避免把历史分数或 diagnostic 当成生产模型证据。任何新方法先补 registry，再进入路线和报告；不允许只在 README 中维护现状。
