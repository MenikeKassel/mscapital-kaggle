# MSCapital 仓库工程化整理报告 (Reorganization Report)

> 日期: 2026-08-15 | 执行: 按用户 20 节 Prompt, Phase A→N
> 核心原则: **先盘点再整理; 保持可运行性第一; 未经确认不删除模型/预测/submission/实验结果; 历史编号保留**

## Before — 主要问题

1. 实验做了 70+ 轮, 但"做过什么/为什么失败/哪些已证伪"只能靠通读 RESULTS.md + 各报告, 无统一索引
2. 文档权威链失同步: 根 README 仍引用 plan-v1.8.0 (实际已 v1.9.0)
3. 无统一实验编号 → 报告/脚本/产物之间的归属关系靠文件名猜测
4. 失败实验无结构化记录 (Do Not Repeat 分散在各报告)
5. 提交记录 (v1-v8b) 有 LB, 但 v9 系列 13 个文件无登记, 无法追溯
6. 根目录散落 arXiv 缓存 html ×11, _lb_tmp LB 快照, catboost_info 训练日志 (无归属)
7. **发现 pre-existing bug**: `src/mscapital/cli.py` merge 冲突错误 — `run-revol-lite` parser 被 `run-m02t` 覆盖 (参数/函数全丢), 测试 1 失败

## Changes — 建立的内容

| 新资产 | 内容 |
|---|---|
| `experiments/registry.csv` | 71 个实验总台账 (ID/阶段/数据/方法/协议/分数/判定/脚本/报告/产物) |
| `experiments/<phase>/<ID>/README.md` | 71 份实验 README (研究问题/Hypothesis/Method/Result/Decision/Failure/DoNotRepeat/Next) |
| `experiments/_tools/` | experiment_data.py (canonical 映射表) + build_registry.py + build_inventory.py (可重跑) |
| `docs/experiment-index.md` | 研究历史总入口 (阶段全表 + 方向汇总 + 状态速览) |
| `docs/failed-experiments.md` | 失败墓地 — 6 类机制分类 (Negative Knowledge Base) |
| `docs/research-findings.md` | 已确认结论 F001~F017 (证据→实验→置信度) |
| `docs/method-map.md` | 方法地图 (✅/🟡/❌/⚠️/🧪) |
| `docs/experiment-inventory.md` | 全仓库资产清单 (脚本 147 + 产物 110 目录 + docs 51 + 其他) |
| `submissions/README.md` | 提交登记表 (v1~v8b 可追溯 + v9 待确认 + P6 候选) |
| 根 README | + Current Research Status 段; plan 引用 v1.8.0→v1.9.0; 路线更新 (幅度关闭/SCFI 确认/P6 待拍板) |
| docs/README.md | + 工程化体系导航节 + P6/P6R/P7/EDA 阶段行 |

## Moved / Renamed — 物理迁移

| 原位置 | 新位置 | 说明 |
|---|---|---|
| `abs_*.html` ×10 + `srch.html` | `_archive/arxiv_pages/` | arXiv 验证缓存 (git mv, 历史保留; 无脚本引用) |
| `_lb_tmp/*.csv` ×7 | `_archive/lb_snapshots/` | LB 历史快照 (99_lb_rank.py 运行时自动重建 _lb_tmp) |
| `catboost_info/` | `_archive/catboost_info/` | CatBoost 训练日志 (gitignore; 训练时自动重建) |

**重命名: 0** (历史编号与文件名全部保留, 未擅自改任何实验脚本/报告名)

## Fixed — 兼容性修复

| 修复 | 详情 |
|---|---|
| `src/mscapital/cli.py` run-revol-lite | 补回被 run-m02t 覆盖的 6 个参数 + func 绑定 (pre-existing merge bug, ff047ac 引入) |
| 验证 | `pytest tests/` 112/112 通过 (修复前 111/112) |

## Deleted

**0 个文件删除。** 所有模型/预测/提交/实验结果原样保留。

## Archived — 归档汇总

`_archive/` 现包含: `plans/` (历史方案 v1.0.0~v1.8.0), `reports/` (早期阶段报告), `arxiv_pages/` (11), `lb_snapshots/` (7), `catboost_info/`, drw 讨论页, 比赛 zip

## Remaining Problems

1. **`docs/dataset-sample-walkthrough-2026-08-15.docx`** — 示例报告 Word 版, 未跟踪, 待用户决定保留/删除
2. **`scripts/p6_candidate_diag.py`** — P6 候选诊断脚本, 未跟踪, 建议入库
3. **`output/submissions/submission_v9_*` (13 文件)** — 无 LB 记录, 需用户确认是否提交过; 若未提交建议移入 `_never_submitted/`
4. **`output/` 41GB** — p5 系列特征 bin 约 30GB 是磁盘清理第一目标 (报告已在 git, bin 可再生成); 本次未动
5. **P6R-01 挂起** — 未拍板未执行 (registry 中标记 SUPERSEDED/未跑)
6. **P6 提交候选** — 未拍板未提交 (登记在 submissions/README.md)

## 验收自检

- [x] 所有正式实验有唯一 Experiment ID (71 个, 历史编号保留)
- [x] 所有实验有状态 (registry.csv status 列: GREEN 30 / YELLOW 12 / RED 25 / INVALID 1 / SUPERSEDED 3)
- [x] 所有实验有 README (71 份)
- [x] 所有失败实验有失败原因 (README + failed-experiments.md 双重)
- [x] 所有关键结论有实验依据 (research-findings.md F001~F017)
- [x] 所有正式 submission 可追溯 (submissions/README.md)
- [x] 不存在 test2/final_new 类命名问题 (历史脚本保留原名, 新脚本规范见 experiments/README.md)
- [x] 实验代码/结果/结论互相可追踪 (registry.csv 三路径列)
- [x] active research direction 一眼可见 (README Research Status + experiment-index 方向汇总)
- [x] 已证伪方向快速查询 (failed-experiments.md + method-map ❌)
- [x] 历史实验未失去复现能力 (0 删除, 0 重命名, 脚本/产物原位; pytest 112/112)
- [x] README 作为项目入口 (含 Research Status + 文档权威链)
- [x] experiment-index.md 作为研究历史总入口
- [x] failed-experiments.md 阻止重复踩坑 (6 类机制 + Do Not Repeat)

## 总结

- 实验总数: **71** (GREEN 30 / YELLOW 12 / RED 25 / INVALID 1 / SUPERSEDED 3)
- 归档文件: 18 (arxiv_pages 11 + lb_snapshots 7 + catboost_info 目录)
- 重命名文件: 0
- 删除文件: 0
- 无法分类文件: 2 (docx 报告 + p6_candidate_diag.py, 均待用户确认)
- 修复 bug: 1 (cli.py run-revol-lite merge 缺陷)

## 维护纪律 (此后)

1. 新实验: `experiments/_tools/experiment_data.py` 加条目 → 重跑 build_registry.py → 提交
2. 新提交: 先登记 `submissions/README.md` → 再提交 Kaggle → 回填 LB
3. 新结论: 入 `research-findings.md`; 新失败: 入 `failed-experiments.md` (机制分类)
4. 阶段完成: 更新 experiment-index.md + RESULTS.md + docs/README.md → git 提交
5. 重跑生成: `python experiments/_tools/build_inventory.py` (inventory 自动更新)
