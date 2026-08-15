# MSCapital – Real Financial Market Forecasting

Kaggle 社区赛的个人研究仓库。任务是根据高频行情（market 600s 盘口）、订单和成交数据预测未来价格变化，评价指标为未中心化 cosine similarity。**数据 = 真实市场数据简化抽样**（官方原话），四文件生成结构见 [docs/data-generation-structure.md](docs/data-generation-structure.md)。

> 研究快照：2026-08-15，Public LB **0.142**（#30/107，v8b）。提交冻结中，新候选必须过本地时序验证 + 分布门禁 + 30 秒审计（见 [docs/calibration.md](docs/calibration.md)）。

# Current Research Status

| 项 | 值 |
|---|---|
| Best Public LB | **0.142** (#30/107) — v8b (v7 + 外部 lb142 推理包 0.5/0.5) |
| Best self-developed | RealMLP 152 特征复刻 (v7 融合 0.8/0.2 → LB 0.135) |
| Current baseline | C-04 frozen: 0.63×RMS(RealMLP) + 0.37×RMS(Clean Table) |
| Current phase | P6 提交候选待拍板 → O→T lag response (GPT P1.5) |
| Active hypothesis | 条件创新 SCFI (O−E[O\|M]) 双 learner 确认有效 (P5-04/D/E) |
| 待拍板 | ① P6 提交 (v8b+0.55×RealMLP-C, 期望 0.1423~0.1428) ② P6R-01 ③ 下一步 |
| Last updated | 2026-08-15 |

→ [Experiment Index](docs/experiment-index.md) · [Research Findings](docs/research-findings.md) · [Failed Experiments](docs/failed-experiments.md) · [Method Map](docs/method-map.md) · [Submission History](submissions/README.md) · [Inventory](docs/experiment-inventory.md)

## 当前结果

| 版本 | 核心变化 | PSEUDO | Public LB | 结论 |
|---|---|---:|---:|---|
| v1–v3 | 官方特征树模型融合 | 0.129–0.130 | 0.122 | 基线 |
| v4 | R2 归一化 | 0.1313 | 0.123 | 跨时期稳定性改善 |
| v5 | 增加 22 个无量纲微观结构特征 | 0.1349 | 0.125 | 稳定正增益 |
| v6 | 表格模型 + TCN（7%） | 0.1379 | 0.082 | 严重 OOD 失败 |
| v7 | v5 + 事件动力学 RealMLP（20%） | 0.1397 | 0.135 | 新表示带来最大自研增益 |
| v8a | v7 + 外部 LB142 预测（30%） | 不可共同验证 | 0.139 | 外部正交 alpha |
| v8b | v7 + 外部 LB142 预测（50%） | 不可共同验证 | **0.142** | 当前最好成绩 |

完整实验流水账见 [RESULTS.md](RESULTS.md)；全部文档导航见 [docs/README.md](docs/README.md)。

## 已确认的关键结论

- 验证必须按月份前向切分；随机 CV 和单一近期 fold 会高估效果。
- 表格特征在 PSEUDO 与榜单之间存在约 `0.008–0.010` 的差距，但不能把这个差值外推到新模型族。
- RealMLP 路线的 v7 PSEUDO 为 `0.139683`，Public LB 为 `0.135`，首个 Regime B 差值为 `0.004683`。
- v7 的 valid/test 预测标准差比为 `0.7447`，说明 test 上存在明显幅度迁移。
- TCN 的 test 预测与表格模型相关性从验证期约 `0.30–0.40` 崩到 `0.03`，即使本地融合增益为正也不能提交。
- 新方法的价值不仅看单模型分数，还要看跨月稳定性、预测尺度和与现有最佳预测的正交性（评估六维：ΔPSEUDO / corr(new,v7) / corr(new,y) / 月命中率 / regime 稳定 / frozen OOS）。
- **cosine loss 是真实但小的互补**（严格复刻 +0.001，两次 Public 0.135 自洽）；**验证命题 ≠ 提交命题**（配方/区间/α 三要素）。
- **market 600s 序列 = 第二套正交 alpha**：MSE 无信号但 cosine 臂 corr(y)=+0.086、corr(market,v7)=0.49（P5-01）；手工聚合形式无信号（34 特征 −0.0002、P4-08 残差链断）→ 信息以序列形式存在。
- 月漂移存在但不可预测（P4-05 决定性负结果）；600s 长上下文解释 LB142 分歧但不解释 y 残差（P4-08 链条断裂）。

## 当前研究路线（plan-v1.9.0 — research.md 四方方案全集收敛）

> **文档权威链（防误读）**: `README.md`（入口/结果快照）→ `docs/plan-v1.9.0.md`（**路线唯一 current source of truth**，历史版本全部归档于 `docs/_archive/plans/`）→ `docs/EXPERIMENT_SUMMARY.md`（实验结果台账）。路线问题一律以 plan-v1.9.0 为准。

已执行并关闭的路线（2026-08-14/15，详见 docs/experiment-index.md）:
- **P5-02M 幅度调制 → P5-03 MAG-Gate RED**（嵌套 Δ −0.000146）+ **P7-01 RED**（GPT1 预注册终裁, α=0 单调）→ 幅度门控方向整体关闭
- **P5-02B-lite → P5-04 SCFI CONTINUE**（LGB Δ+0.0075）→ **P5-06/E 生产确认**（RealMLP standalone +0.0040 / blend +0.0014 双窗口一致）→ **P6 生产推理完成, 提交候选就绪（未提交, 等拍板）**
- **P5-05 RICS / SPCE 谱线 RED**（短窗形态 ≤0.011）→ 相位/形状线关闭（TRIS 仅保留 random 臂待定）
- **P6R-00 检索残差 KILL**（gate 39%）, P6R-01 终裁挂起

当前队列（plan v1.9.0 §3）:
1. **O→T lag response（GPT P1.5, 推荐下一步）** — 事件级跨表时序, 152 特征确认无覆盖, 与 P5-02 "跨通道同步是核心" 同源
2. **P6 提交候选拍板**（v8b + 0.55×RealMLP-C, 双窗口门禁已过）
3. 后置: ETCI 事件时臂（与 lag response 同源）/ hard regime experts（需重新论证）/ soft MoE / 条件模型完整版（FiLM, 等 SCFI 消化）
4. 停止（继承）: TCN 调参 / Transformer / 152 cosine 变体 / RealMLP 搜索 / 换皮 / calibration / ensemble 套娃 / **幅度 gate 任何变体** / **残差均值-检索路线** / 不经审计直接展开交互或条件模型

**market 序列建模口径（已统一）**: 200 步 × **18 通道**（11 raw + 7 derived: mid/spread1/depth1/imb1/spread2/depth2/imb2），均匀 3s 网格。⚠️ 历史文档中的 "13ch" 是 P5-01 之前的旧规划残留（当时计划 11 raw + 2 derived），实际实现一律 18ch；market.feather 原始表 13 列 = schema 列数（sample_id + seconds + 11 盘口字段），不是建模通道数。

## 仓库结构

```text
.
├── scripts/                  # 特征、验证、模型、融合和 Kaggle kernel 脚本（含 p4_*/p5_* 系列）
├── src/mscapital/            # 可复用库代码（realmlp.py 等）
├── tests/                    # 关键运行管线的回归测试 (112 用例)
├── experiments/              # 实验逻辑层: registry.csv (71 实验) + 每实验 README (规范 ID/判定/失败分析)
├── submissions/              # 提交登记表 (LB ↔ 脚本 ↔ 实验 ID 可追溯)
├── docs/
│   ├── README.md             # 文档索引（导航入口）
│   ├── experiment-index.md   # 研究历史总入口 (5 分钟看懂全项目)
│   ├── research-findings.md  # 已确认结论 (F001~F017)
│   ├── failed-experiments.md # 失败墓地 (按机制分类, Negative KB)
│   ├── method-map.md         # 方法地图 (✅/🟡/❌/🧪)
│   ├── experiment-inventory.md  # 全仓库资产清单
│   ├── reorganization-report.md # 本整理迁移报告
│   ├── plan-v1.9.0.md        # 当前方案（唯一 current source of truth, 旧版在 _archive/plans/）
│   ├── EXPERIMENT_SUMMARY.md # 当前实验总览与决策（结果台账）
│   ├── data-generation-structure.md  # 四文件生成结构取证 + 最像比赛
│   ├── calibration.md        # 分模型族校准与提交门禁
│   └── _archive/             # 历史方案/阶段报告/抓取快照
├── research/                 # 方法库（METHODS.md, literature_primer.md）
├── configs/                  # 实验配置
├── output/                   # 训练产物（预测/权重/特征, gitignored, ~41GB）
├── _archive/                 # 仓库杂物归档（arxiv 页面缓存/LB 快照/catboost_info）
└── RESULTS.md                # 详细实验日志和负面结果库
```

原始数据（`D:\mscapital-forecasting\data\raw`）、加工特征、模型权重、预测文件和提交文件均不进入 Git。特征缓存 parquet 经 hardlink 去重（三份路径共享同一 inode，改任一路径会破坏链接）。归档的旧 worktree 在 `D:\mscapital-forecasting\archive-worktrees\`。

## Git 纪律（2026-08-14 整理后）

- 工作分支 = **main**（已快进合并 p4-hidden-info 全部 18 提交）；所有历史实验分支已删除，**tag `archive/<分支名>` 备份**（`git tag -l 'archive/*'` 可查）。
- 每阶段实验（实验链/提交批次/决策门）完成 → 更新 RESULTS.md + docs/README.md → 提交。
- 只提交代码与文档；数据/权重/预测/提交文件一律 gitignore（规则见 `.gitignore`）。

## 环境

- Python 3.11（`.venv`，Windows）
- Polars / pandas / PyArrow；LightGBM / XGBoost / CatBoost / scikit-learn；PyTorch 2.x（本地 CUDA 与 Kaggle P100 分别处理，见 skill）
- 运行脚本前 `export PYTHONPATH=`（避免环境变量污染 venv）

部分脚本是实验快照，路径参数仍与本地数据布局绑定。复现实验前应先阅读 [docs/calibration.md](docs/calibration.md) 和对应脚本顶部的配置。

## 数据与安全

- 不提交 Kaggle 原始数据、派生 parquet、模型权重或 submission CSV。
- 凭证只通过环境变量或 Kaggle Secrets 注入，不写入代码和文档。
- 外部公开方案的预测仅作为 ensemble probe 管理，不进入本地分数校准模型。
