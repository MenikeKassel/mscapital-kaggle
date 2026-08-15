# MSCapital Experiment ID 迁移审计报告 (2026-08-15)

> 任务书: 48 节 Experiment ID v1.0 工程化重构 | 执行: Phase A→L + §42-45
> 判定: **GREEN** (验收 27 项全过, pytest 138/138)

## 1. Before Statistics (Phase A 真实审计, 脚本统计)

| 指标 | 值 |
|---|---|
| total_registry_entries | 71 |
| canonical_valid (v1 语法) | 0 (无 id_status 列, 全部未分类) |
| ID 形态数 | 9 种 (P{n}-NN / P5-A~E / P0.5-X / P6R-NN / P7-AMP / M0x / SUB-vx / C1~C4 / E01~03) |
| 历史重名 (已在前轮修复) | B1, C1, E1, H1 |
| 数据缺口 (本轮发现) | P4-08B 漏录, M02-T 漏录, P4-02a/b/c 被错误合并, S-01/02/03/06 提交事件缺失 |

## 2. Final ID Specification

见 [experiment-id-spec-v1.0.md](./experiment-id-spec-v1.0.md) (已冻结)。

## 3. Canonical Namespace

```
^(P[0-7]|P6R|C|E|M|S)-[0-9]{2}[a-z]?$
```
67 个 canonical entries (见 registry.csv)。

## 4. Legacy Allowlist

15 个冻结条目: `B0 B1 A1 A2 B1-LGO B2 C1-FE D1 E1-TW F1 F2 G1 G2 G3 H1`
- 依据: baseline 阶段早期表格实验, 已进入 v1-v3 提交链与历史报告; 不重编号
- 封闭: `tests/test_registry_consistency.py::test_legacy_allowlist` 强制

## 5. Alias Rules

- 82 条记录, 31 个 alias (旧 ID 全部保留, | 分隔)
- 全局唯一 ✓ 无 chain ✓ 无 primary 冲突 ✓ (pytest 断言)
- 例: `P5-A|MAG-Gate → P5-03`, `P7-AMP → P7-01`, `SUB-v8 → S-08`, `P0.5-C → P0-05`, `M02-T → M-03`

## 6. P4-06A Adjudication (Phase B)

| 项 | 判定 |
|---|---|
| 旧 ID | P4-06A (600s long-context residual probe) |
| 分类 | **INDEPENDENT** (独立 hypothesis: 残差跨窗可复用性; 独立 protocol: short/long/both; 独立结果: 链条断裂) |
| 迁移 | P4-06A → **P4-08** (alias: P4-06A) |

## 7. P4-08A~E Adjudication (Phase B, 任务书 blocker)

逐个读取 README/script docstring/output json/报告 §12-14 证据:

| 旧 ID | 脚本 | 独立 hypothesis? | 独立结果? | 分类 | 迁移 |
|---|---|---|---|---|---|
| P4-08A | p4_08a_loss_ablation.py | 训练目标错配假设 | cosine +0.00703 vs MSE 0 | **INDEPENDENT** | → P4-10 |
| P4-08B | p4_08a_unc.py + 48b_realmlp_cosine_prod.py | 生产 loss 配置层错配 (lambda_cos 0.01→1.0) | 生产预测生成 | **INDEPENDENT** (补录) | → P4-11 |
| P4-08C | p4_08c_blend.py | blend → submission | v9_cos 提交候选 | **INDEPENDENT** | → P4-12 |
| P4-08D | p4_08d_simple_cosine_prod.py | simple-MLP 全量生产 | 生产预测 | **INDEPENDENT** | → P4-13 |
| P4-08E | p4_08e_v7like_check.py | 融合对象不同 → 增益不同 | +0.00098 (v7_like) | **INDEPENDENT** | → P4-14 |

**同时裁决**: P4-02a/b/c (factors 逆向 / market-forms / OFI protocol) 三个独立实验被错误合并为 P4-02 → 拆分 → P4-02/P4-03/P4-04。

## 8. Full Migration Map

见 [experiment-id-migration-map-2026-08-15.csv](./experiment-id-migration-map-2026-08-15.csv) (82 行: old_id/new_id/id_status/phase/parent/reason/title/decision/status)。
代表性迁移:
- `P0.5-B/C/D → P0-04/P0-05/P0-06` (点号废弃)
- `P5-A/B/C/D/E → P5-03~P5-07` (独立序号)
- `P5-02I → P5-02` (大小写统一)
- `P4-H1H2/P4-LB142/P4-MH → P4-15/P4-16/P4-17` (描述词→序号)
- `P7-AMP → P7-01`; `P6 → P6-01`; `P2 → P2-01`
- `C1~C4 → C-01~C-04`; `E01~E03 → E-01~E-03`; `M01-A~M06 → M-01~M-07` (M02-T 补录 → M-03)
- `SUB-v4/5/7/8 → S-04/S-05/S-07/S-08`; 补录 `S-01/02/03/06`

## 9. Registry Schema Changes

v1 (20 列) → v2 (28 列):
```
+ id_status (canonical/legacy)      + status (lifecycle 6 态)
+ decision (GREEN/YELLOW/RED/NA)    + parent / successor (谱系)
+ title (中文) + name (ASCII slug)  + aliases (| 分隔)
+ tags (检索)                       + created_at
```
meta: `experiments/registry.meta.json` (schema_version=2, spec_version=1.0)

## 10. Allocator Design

`experiments/_tools/new_experiment.py`:
- `new_experiment.py P5 market-volatility-residual [--parent P5-01]`
- `new_experiment.py --arm P5-09 cosine` (arm 只能在存在 canonical parent 下)
- monotonic max+1 (不回填空洞), ID 不可复用, 注册即 planned + 目录/README/results/logs 骨架
- 演练 (§42): 临时 registry 验证 P5-04 分配 (跳过 P5-02 空洞) / arm P5-01a / 拒绝 P8/P5R/TEMP — PASS, 生产未污染

## 11. Resolver Design

`src/mscapital/experiment_registry.py::resolve_experiment_id()`:
- canonical 解析自身; alias 一步解析; legacy 解析自身; 未知 → KeyError
- 全库唯一映射源 (build_alias_index), 不再维护独立映射表

## 12. Lifecycle Design

```
planned → running → completed → (superseded | deprecated | aborted)
decision 独立: GREEN/YELLOW/RED/NA
```
状态机由 registry 列 + 测试强制 (planned/running ⇒ NA; completed ⇒ 有裁决)。

## 13. New Experiment Workflow

```
Idea → Hypothesis → new_experiment.py → Canonical ID → Registry: planned
→ Execution → Registry: running → Result → Decision → Report
→ Registry: completed → Parent/Successor → Lineage
```
No ID, No Formal Experiment — 未经注册的内容只算 scratch, 不得进入正式结论/submission。

## 14. Consistency Test Coverage

`tests/test_registry_consistency.py` 26 用例: ID 语法/legacy allowlist/唯一性/series 封闭/P6R 特殊/alias 全局唯一·无冲突·无 chain·单步/resolver 全路径/allocator monotonic·不复用·arm 门禁·非法 series/lifecycle 枚举·分离·RED 必含 failure/目录·README 存在·无 orphan·script 存在/schema meta。**改坏编号 = 测试红**。

## 15. After Statistics

| 指标 | Before | After |
|---|---|---|
| entries | 71 | **82** (+11: P4-02 拆分 +2, P4-08B +1, M02-T +1, S 提交 +4, 其他) |
| canonical | 0 (未分类) | 67 |
| legacy | 0 (未分类) | 15 |
| invalid_unclassified | 71 | **0** |
| alias_collision | - | 0 |
| orphan directories | 0 | 0 |
| 目录命名 | 中文截断 | `<ID>_<ASCII slug>` |
| pytest | 112 | **138** (26 新 consistency) |

## 16. Remaining Risks

1. **历史报告正文保留旧 ID** (EXPERIMENT_SUMMARY/exploration-report/gpt-review 等): 有意为之 (证据不可篡改), 检索旧 ID 时需经 resolver
2. **S-08 覆盖 v8/v8b 两个提交** (0.5 权重变体): 未来若需区分可加 arm (S-08a/b)
3. **P4-11/12/13 (cosine 生产系) 的 v9 提交候选无 LB 记录**: 与 submission 登记表一致, 待用户确认是否提交过
4. **baseline legacy 区不做语法校验**: 15 个 ID 豁免正则, 但由 allowlist 封闭
5. 旧工具 `experiments/_tools/fix_ids.py` / `update_docs_ids.py` 为一次性脚本, 保留作审计记录

## 17. pytest Results

```
138 passed (17.2s) — tests/ 全量 (112 历史 + 26 registry consistency)
```

## 18. Final Decision

**GREEN** — Experiment ID Specification v1.0 正式冻结; 所有红线 (任务书 §40 R1-R8) 由测试强制; 后续新实验全部经 allocator 进入体系。
