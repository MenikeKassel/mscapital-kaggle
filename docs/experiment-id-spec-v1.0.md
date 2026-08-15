# MSCapital Experiment ID Specification v1.0 (冻结)

> 冻结日期: 2026-08-15 | 适用: 本仓库全部正式实验 (当前 82 条)
> 状态: **FROZEN** — 此后不得进行第三轮人工重编号; 修改本规范必须: 改 schema + 改测试 + 更新文档 + 工程评审
> 执行原则: **No ID, No Formal Experiment** · **Canonical IDs are immutable and never reused** · **Independent hypotheses receive independent numeric IDs** · **Arms are not experiments** · **Legacy is frozen** · **Registry is the single source of truth**

## 1. Canonical ID 语法

```
^(P[0-7]|P6R|C|E|M|S)-[0-9]{2}[a-z]?$
```

- 合法 series (封闭集合): `P0 P1 P2 P3 P4 P5 P6 P6R P7 C E M S`
- 两位数字序号 = 阶段内分配序 (allocator 分配, 非人工)
- 小写后缀 `a/b/c` = 仅同一实验内部 arms (受控变体)
- 禁止: `P8-*`, `P5R-*`, `P5-A`, `P7-AMP`, `P0.5-*`, `TEMP-*`, `FINAL-*`, `TEST-*`

## 2. 特殊 series

- `P6R` 是历史形成且仍有效的特殊 series — **非开放模板**, 不得衍生 P5R/P7R/P6A
- 新增特殊 series 必须: 修改 schema + 修改测试 + 更新规范 + 工程评审

## 3. ID 语义边界

- **独立实验** (独立 hypothesis/protocol/result/conclusion) → 独立数字 ID
- **arm** (同一设计内的 controlled arms/ablations) → 小写后缀
- **关系** 由 `parent` / `successor` 表达, 不靠编号
- arm 产生新研究问题 → 新数字 ID + `parent` 指向原实验

## 4. Legacy 冻结区

`LEGACY_ALLOWLIST` (frozen, 禁止新增):
```
B0 B1 A1 A2 B1-LGO B2 C1-FE D1 E1-TW F1 F2 G1 G2 G3 H1
```
- legacy 是封闭历史集合, 不是垃圾桶; 新实验不得以 `id_status=legacy` 绕过规则

## 5. Alias 规则

- alias 全局唯一; 不得与 canonical/legacy primary ID 冲突; 一步解析 (禁止 chain); target 必须存在
- 历史报告正文保留旧 ID (证据不可篡改); registry 负责关联

## 6. Allocator 规则

- 唯一正式入口: `python experiments/_tools/new_experiment.py <series> <slug> [--parent X] [--arm ...]`
- monotonic: `next = max(ever_allocated) + 1`; 不回填空洞; ID 永不回收/复用
- 分配即写入 registry (`status=planned`, `decision=NA`) + 创建目录/README/config 骨架

## 7. 生命周期状态机

```
status ∈ {planned, running, completed, aborted, deprecated, superseded}
decision ∈ {GREEN, YELLOW, RED, NA}
```
- status 与 decision 分离: `completed + RED` 合法且重要 (负结果)
- planned/running 必须 decision=NA

## 8. Registry 即 SSOT

- `experiments/registry.csv` (schema v2) + `experiments/registry.meta.json` (schema_version=2, spec_version=1.0)
- 目录名/README/索引/lineage 全部由 registry 派生; 禁止分头维护
- 一致性由 `tests/test_registry_consistency.py` (26 用例) 强制

## 9. 新实验流程 (强制)

```
Idea → Hypothesis → new_experiment.py → Canonical ID → Registry(planned)
→ Execution → Registry(running) → Result → Decision → Report
→ Registry(completed) → Parent/Successor → Lineage
```
- 未经 registry 登记的内容 = scratch, 不得进入正式结论/submission
- scratch 值得继续 → 必须通过 allocator 重新注册后按正式 protocol 执行
