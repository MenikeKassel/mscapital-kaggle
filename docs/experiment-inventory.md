# MSCapital 实验资产清单 (Experiment Inventory)

> 生成: 2026-08-15 (仓库工程化整理 Phase A) | 重跑: `python experiments/_tools/build_inventory.py`
> 角色: 全仓库资产地图 — 判断每个文件是什么/属于哪个实验/是否在用

## 0. 顶层结构

```
drwxr-xr-x 1 menike 197121 0M  8月 15 15:05 .
drwxr-xr-x 1 menike 197121 0M  8月 14 15:19 ..
-rw-r--r-- 1 menike 197121 1M  8月 13 01:19 .gitignore
drwxr-xr-x 1 menike 197121 0M  8月 12 22:18 .pytest_cache
drwxr-xr-x 1 menike 197121 0M  8月 15 15:01 _archive
drwxr-xr-x 1 menike 197121 0M  8月 15 15:01 _lb_tmp
drwxr-xr-x 1 menike 197121 0M  8月 13 15:37 configs
drwxr-xr-x 1 menike 197121 0M  8月 15 15:05 docs
drwxr-xr-x 1 menike 197121 0M  8月 15 15:01 experiments
drwxr-xr-x 1 menike 197121 0M  8月 10 21:42 notebooks
drwxr-xr-x 1 menike 197121 0M  8月 15 14:27 output
-rw-r--r-- 1 menike 197121 1M  8月 13 15:37 pyproject.toml
-rw-r--r-- 1 menike 197121 1M  8月 14 17:06 README.md
drwxr-xr-x 1 menike 197121 0M  8月 14 22:14 research
-rw-r--r-- 1 menike 197121 1M  8月 15 14:43 RESULTS.md
drwxr-xr-x 1 menike 197121 0M  8月 15 14:46 scripts
drwxr-xr-x 1 menike 197121 0M  8月 12 22:19 src
drwxr-xr-x 1 menike 197121 0M  8月 15 15:05 submissions
drwxr-xr-x 1 menike 197121 0M  8月 14 17:34 tests
```

## 1. scripts/ — 实验脚本 (147 个 .py + 9 个子目录)

按编号段分类:

| 段 | 实验归属 | 文件数 |
| -- | -- | -: |
| 00-18 | baseline 表格阶梯 (B/A/C/D/E/F/G/H + v1/v2) | 00_verify_data.py 02_cache_features.py 03-18 |
| 19-26 | P0 协议 (Temporal Matrix/AV/R2/v3/v4) | 19-26 |
| 27-38 | P1 微观特征/TCN (P1-1a/b/c/e, P1-2, v5/v6) | 27-38 |
| 40-48 | P1-P2 生产 (0726/RealMLP/lb142/v7/v8, C 系列) | 40-48* |
| 99_* | 工具类 (arxiv 验证/kaggle kernel 管理/调试) | 99_*.py |
| p3_* | P3 下一代方法 (SAE/条件化/掩码/网格/NHP) | p3_*.py |
| p4_* | P4 隐藏信息调查 (01a~08E/H1H2/LB142/MH) | p4_*.py |
| p5_*/p5?_* | P5 市场探针 (01/02I/A/B/C/D/E) | p5*.py |
| p6_*/p6r_*/p7amp_* | P6 生产/P6R 检索/P7 幅度 | p6*.py p6r_*.py p7amp_*.py |
| m0* | M 系列 (经 src/mscapital CLI) | m01a_* m02_* m03_* m04_* m05_* m06_* |
| e0* | E 系列 (经 src/mscapital CLI) | e01_* e02_* e03_* |
| eda_* | EDA 与示例报告 | eda_*.py |
| diag_* | 诊断 (幅度/特征清单) | diag_*.py diag_amplitude.py |
| b1_* | 官方基线 | b1_official_baseline.py |
| kaggle_* (dir) | Kaggle 云端 kernel 包 (9 个) | kaggle_* |
| build_kaggle_* | 云端 kernel 构建 | build_kaggle_c1/c3.py |
| 其他 | monitor/ps_procs/rebuild/smoke/zz_* | monitor_kaggle.py ps_procs.ps1 rebuild_p5_02i_json.py smoke_p5_02i.py zz_*.py |

> 每个实验的脚本归属见 experiments/registry.csv `script_path` 列。

## 2. src/mscapital/ — 项目库 (正式 CLI + 模型/特征)

| 模块 | 角色 |
| -- | -- |
| cli.py | 全部 CLI 命令 (baseline/C 系列/M 系列/E 系列/retrieval) |
| models/ | realmlp, clean_table, m01a~m06, context_shift, revol_lite, residual_catboost, retrieval |
| features/ | event_flow, lob_geometry, geometry_temporal, ofi, optiver_interactions, path_signature, revol_lite |
| metrics.py / splits.py / stability.py | cosine 指标 / 时序切分 / 稳定性审计 |
| artifacts.py / config.py / diagnostics.py / preprocessing.py / residual.py / ensemble.py | 产物校验 / 配置 / 诊断 / 预处理 / 残差 / 融合 |

## 3. tests/ — 13 个测试文件 (112 用例, 全部通过)

覆盖: clean baseline / RealMLP builder / retrieval 时序断言 / residual 泄漏 / protocol-v2 / path signature / revol-lite CLI。

## 4. docs/ — 报告与文档 (51 文件)

| 类别 | 文件 |
| -- | -- |
| 导航/权威 | README.md (索引), experiment-index.md (研究历史总入口), plan-v1.9.0.md (唯一 current), RESULTS.md (台账), EXPERIMENT_SUMMARY.md, exploration-report.md |
| 本整理产出 | experiment-inventory.md, failed-experiments.md, research-findings.md, method-map.md, reorganization-report.md |
| 协议 | calibration.md, protocol-v2.md, canonical-residual-oof.md, p6r_preregistration.md, p6r_repo_audit.md |
| Baseline/P0-P2 | c1~c4 系列 (8), p3-results.md, e01-e02-e03-results.md, m01~m06 (8) |
| P4 | p4-hidden-information-report.md, p4-01a-market-forensics-report.md |
| P5 | p5-02i-info-audit-report.md, p5a-mag-gate-report.md, p5b-scfi-report.md, p5c-rics-report.md, p5-final-decision.md, p5de-production-verification.md |
| P6/P7 | p6-production-inference.md, p6r_experiment_report.md, p7amp-quick-results.md |
| EDA/评审 | eda-raw-2026-08-15.md, eda-vs-pipeline-2026-08-15.md, dataset-sample-walkthrough-2026-08-15.md, gpt-eda-review/gpt1-round2-review/gpt-review-p5-round (3) |
| 方法调研 | next-gen-methods.md, similar-competitions.md, method-provenance.md, method-transfer-sprint.md, remaining-methods-summary.md, data-generation-structure.md |
| 归档 | _archive/ (plans/, reports/, arxiv_pages/, lb_snapshots/, catboost_info/) |

## 5. output/ — 实验产物 (41GB, gitignore)

| 族 | 内容 | 说明 |
| -- | -- | -- |
| c1/c2/c3/c4_* | C 系列产物 (18 目录) | formal/local/compare/protocol_closed/scale 等变体 |
| kaggle_c* / kaggle_rolling_* | 云端 kernel 下载 (12 目录) | P100 产物回传 |
| p3_* | P3 产物 (8 目录) | sae/conditioned/masked/grid/nhp |
| p4_* | P4 产物 (18 目录) | 01a/02/03/04/05/06a/07/08a/forensics/lb142/market_history |
| p5_01/p5_02i/p5a/p5b/p5c/p5d/p5e | P5 产物 (7 目录, 30GB) | 特征 bin 大文件 (p5c 7.6G 等) |
| p6_prod / p6r_00 | P6 生产 / P6R 检索 | submission_candidate_p6.csv 在此 |
| m01a~m05_* | M 系列 (13 目录) | features + formal |
| e01/e02/e03 | E 系列 (5 目录) | revol/context/stability |
| p12* | TCN 系列 (7 目录) | kernel 日志 + npz |
| rlps_* | RealMLP PSEUDO (9 目录) | v12 为最终产物 |
| canonical_* | canonical OOF (2 目录) | blocks + residual npz |
| f0726_* | 152 特征 kernel (3 目录) | 本地训练缓存 |
| submissions | 30 个提交 csv (540MB) | 登记见 submissions/README.md |
| smoke_*/local_temp/logs/experiments | smoke 测试/临时 (5 目录) | 可清理候选 |
| 根散落 | mlp_seed*.pt ×4, temporal_monthly_curves.csv, monthly_target_diagnostics.csv, m06_audit.json | 早期模型权重/诊断 |

> ⚠️ 大文件提示: p5 系列特征 bin 共 ~30GB, 是未来磁盘清理的第一目标; 删除前先确认对应报告已入库 (报告在 git, bin 可再生成)。

## 6. 其他区域

| 区域 | 内容 |
| -- | -- |
| research/ | 方法调研: breakthrough-top3, METHODS.md, literature_primer, new-methods-scout, arxiv_*.json, paper_cards/, tmp/ (网页缓存) |
| configs/ | 9 个 json: c2 消融 5 + clean-baseline/realmlp/table + m01-a |
| notebooks/ | 空目录 (比赛未用 notebook) |
| _archive/ | drw 讨论页, 比赛 zip, arxiv_pages/ (11), lb_snapshots/ (7), catboost_info/, plans/, reports/ |
| 根目录 | README.md, RESULTS.md, pyproject.toml, .gitignore |
| experiments/ | 本整理建立: registry.csv + 71 个实验 README (逻辑层, 不入物理路径) |
| submissions/ | 本整理建立: README.md 登记表 (逻辑层) |

## 7. 未分类/待处理

- `docs/dataset-sample-walkthrough-2026-08-15.docx` — 示例报告的 Word 版 (未跟踪, 待用户决定保留/删除)
- `scripts/p6_candidate_diag.py` — P6 候选测试集诊断脚本 (未跟踪, 建议入库)
- `output/submissions/submission_v9_*` (13) — 无 LB 记录, 待确认是否提交过
- `output/smoke_*`, `output/local_temp`, `output/logs` — 临时产物, 可清理