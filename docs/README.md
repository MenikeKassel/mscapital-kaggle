# MSCapital 项目文档索引 (docs/)

> 维护: 每阶段实验后更新本索引 + git 提交。
> **文档权威链**: `README.md`（入口/结果快照）→ `plan-v1.9.0.md`（路线唯一 current source of truth）→ `EXPERIMENT_SUMMARY.md`（结果台账）。历史 plan 全部归档于 `_archive/plans/`，不保留在 docs/ 根目录，避免状态歧义。

## 🧭 工程化体系 (2026-08-15, 研究历史入口)

| 文档 | 角色 |
|---|---|
| [experiment-index.md](./experiment-index.md) | **研究历史总入口** — 按阶段全表 + 方向汇总 (5 分钟看懂) |
| [experiment-id-spec-v1.0.md](./experiment-id-spec-v1.0.md) | **Experiment ID 规范 v1.0 (冻结)** — 语法/alias/allocator/lifecycle 红线 |
| [experiment-id-migration-audit-2026-08-15.md](./experiment-id-migration-audit-2026-08-15.md) | 编号体系迁移审计报告 (18 节, GREEN) |
| [experiment-id-migration-map-2026-08-15.csv](./experiment-id-migration-map-2026-08-15.csv) | 82 行迁移映射 (old→new) |
| [experiment-lineage.md](./experiment-lineage.md) | 研究谱系树 (由 registry parent 自动生成) |
| [research-findings.md](./research-findings.md) | 已确认结论 F001~F017 (证据→实验 ID→置信度) |
| [failed-experiments.md](./failed-experiments.md) | 失败墓地 (按 6 类机制分类, Negative Knowledge Base) |
| [method-map.md](./method-map.md) | 方法地图 (✅/🟡/❌/⚠️/🧪 全状态) |
| [experiment-inventory.md](./experiment-inventory.md) | 全仓库资产清单 (可重跑生成) |
| [reorganization-report.md](./reorganization-report.md) | 首轮整理迁移报告 (2026-08-15) |
| [../experiments/registry.csv](../experiments/registry.csv) | **总实验台账 SSOT** (82 实验, schema v2, 机器可读) |
| [../submissions/README.md](../submissions/README.md) | 提交登记表 (LB 可追溯, S-01~S-08) |

## 📋 当前方案与总览

| 文档 | 角色 |
|---|---|
| [plan-v1.9.0.md](./plan-v1.9.0.md) | **当前方案 (唯一 current source of truth)** — v1.9.0: research.md 四方方案全集收敛 (GPT1+Grok1+deepseek+GPT2) |
| [EXPERIMENT_SUMMARY.md](./EXPERIMENT_SUMMARY.md) | 实验总览汇总 (2026-08-14) — 结果台账 |
| [exploration-report.md](./exploration-report.md) | 探索全景报告 (2026-08-14) — 方法/原理/效果/失败原因, 覆盖 07-07 起全部尝试 |
| [data-generation-structure.md](./data-generation-structure.md) | 四文件数据生成结构取证 + 最像比赛 (TATC/DRW) + 最优方案 (2026-08-14) |
| [calibration.md](./calibration.md) | CV/PSEUDO/LB 校准表 |
| [protocol-v2.md](./protocol-v2.md) | Protocol-v2 残差验证管线 |
| [canonical-residual-oof.md](./canonical-residual-oof.md) | canonical residual OOF 说明 |

## 🔬 实验记录 (按阶段)

| 阶段 | 文档 |
|---|---|
| C 系列 (Clean Baseline) | [c1-clean-realmlp.md](./c1-clean-realmlp.md) / [results](./c1-clean-realmlp-results.md) · [c2-ablation](./c2-realmlp-ablation.md) / [results](./c2-realmlp-results.md) · [c3-clean-table](./c3-clean-table.md) / [results](./c3-clean-table-results.md) · [c4-baseline-results](./c4-clean-baseline-results.md) |
| M 系列 (M01-M06 表示) | [m01-event-flow](./m01-a-event-flow.md) / [results](./m01-a-results.md) · [m02-geometry](./m02-geometry.md) / [results](./m02-geometry-results.md) · [m02-t](./m02-t.md) / [results](./m02-t-results.md) · [m03-path-signature-results](./m03-path-signature-results.md) · [m04-optiver-interactions-results](./m04-optiver-interactions-results.md) · [m05-market-state-results](./m05-market-state-results.md) · [m06-cross-sectional-audit](./m06-cross-sectional-audit.md) |
| P3 (二代表示) | [p3-results.md](./p3-results.md) (SAE/掩码预训练/网格 — 全部 gate F 或终止) |
| P4 (隐藏信息调查) | [p4-hidden-information-report.md](./p4-hidden-information-report.md) (权威报告) · [p4-01a-market-forensics-report.md](./p4-01a-market-forensics-report.md) (三门禁 CONDITIONAL GO) |
| E 系列 | [e01-e02-e03-results.md](./e01-e02-e03-results.md) (状态条件化 E-02 gate True) |
| P5 Probe 三件套 | [p5a-mag-gate-report.md](./p5a-mag-gate-report.md) (MAG KILL) · [p5b-scfi-report.md](./p5b-scfi-report.md) (SCFI CONTINUE, LGB 强/NN 无) · [p5c-rics-report.md](./p5c-rics-report.md) (RICS KILL) · **[p5-final-decision.md](./p5-final-decision.md) (三 Probe 汇总裁决)** · [p5de-production-verification.md](./p5de-production-verification.md) (P5-06/E: SCFI 生产级确认, RealMLP Δ+0.0040) |
| P6 / P6R / P7 (生产与终裁) | [p6-production-inference.md](./p6-production-inference.md) (提交候选就绪未提交) · [p6r_experiment_report.md](./p6r_experiment_report.md) (检索残差 KILL) · [p7amp-quick-results.md](./p7amp-quick-results.md) (幅度门控 RED) |
| EDA 与评审 | [eda-raw-2026-08-15.md](./eda-raw-2026-08-15.md) · [eda-vs-pipeline-2026-08-15.md](./eda-vs-pipeline-2026-08-15.md) · [dataset-sample-walkthrough-2026-08-15.md](./dataset-sample-walkthrough-2026-08-15.md) (4 真实样本) · [gpt-eda-review-2026-08-15.md](./gpt-eda-review-2026-08-15.md) · [gpt1-round2-review-2026-08-15.md](./gpt1-round2-review-2026-08-15.md) · [gpt-review-p5-round-2026-08-15.md](./gpt-review-p5-round-2026-08-15.md) |

## 📚 方法调研与溯源

| 文档 | 角色 |
|---|---|
| [next-gen-methods.md](./next-gen-methods.md) | 下一代方法 Top 5 决策报告 |
| [similar-competitions.md](./similar-competitions.md) | 类似比赛 Top Solutions 矩阵 |
| [method-provenance.md](./method-provenance.md) | 方法溯源审计 (论文/启发式/内部假设三分类) |
| [method-transfer-sprint.md](./method-transfer-sprint.md) | 方法迁移 sprint |
| [remaining-methods-summary.md](./remaining-methods-summary.md) | 剩余方法总结 |

## 🗄️ 归档 (历史版本, git 历史可追溯)

- `_archive/plans/` — plan-v1.0.0 ~ v1.7.0 (方案历史版本全部归档; docs/ 根目录只保留当前版 plan-v1.8.0.md)
- `_archive/reports/` — project_report_v1~v3 (早期阶段报告, 给 Codex/GPT 评审用) · handoff_codex.md (Codex 交接压缩文档)

## 命名与维护纪律

- 方案文档按 `plan-vX.Y.Z.md` 版本化递增, 文件名与版本号对齐
- 结果文档 `mXX-<name>-results.md` / `pXX-<name>.md`, 附权威报告指针
- 每个阶段完成 (实验链/提交批次/决策门) 后: 更新 EXPERIMENT_SUMMARY + 本索引 + git 提交
- 外部评审 (GPT/Codex) 意见 → `plan-vX.Y.Z.md` 版本历史记录, 不单独建文件
