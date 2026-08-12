# MSCapital – Real Financial Market Forecasting

Kaggle 社区赛的个人研究仓库。任务是根据高频行情、订单和成交数据预测未来价格变化，评价指标为未中心化 cosine similarity。

> 研究快照：2026-08-12，Public LB **0.142**，当时排名 **#30/107**。提交暂时冻结，所有新候选必须先通过本地时序验证与分布门禁。

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

完整实验流水账见 [RESULTS.md](RESULTS.md)，压缩后的实验归因与下一步见 [docs/EXPERIMENT_SUMMARY.md](docs/EXPERIMENT_SUMMARY.md)。

## 已确认的关键结论

- 验证必须按月份前向切分；随机 CV 和单一近期 fold 会高估效果。
- 表格特征在 PSEUDO 与榜单之间存在约 `0.008–0.010` 的差距，但不能把这个差值外推到新模型族。
- RealMLP 路线的 v7 PSEUDO 为 `0.139683`，Public LB 为 `0.135`，首个 Regime B 差值为 `0.004683`。
- v7 的 valid/test 预测标准差比为 `0.7447`，说明 test 上存在明显幅度迁移。
- TCN 的 test 预测与表格模型相关性从验证期约 `0.30–0.40` 崩到 `0.03`，即使本地融合增益为正也不能提交。
- 新方法的价值不仅看单模型分数，还要看跨月稳定性、预测尺度和与现有最佳预测的正交性。

## 当前研究路线

1. 逆向归因 LB142 精简包，明确五个 v9 成员、v10 factors 和 grid 表示的真实信息来源。
2. 构建 Residualized Dynamics V2：用多尺度变化、事件时间速度、路径形态和流动性恢复特征预测 v7 的 OOF 残差。
3. 验证严格历史邻居的 Market-State KNN，优先预测现有模型残差。
4. 小规模验证低阶 Path Signature；不恢复 Transformer/TCN 架构搜索。

## 仓库结构

```text
.
├── scripts/                  # 特征、验证、模型、融合和 Kaggle kernel 脚本
├── tests/                    # 关键运行管线的回归测试
├── docs/
│   ├── EXPERIMENT_SUMMARY.md # 当前实验总览与决策
│   ├── calibration.md        # 分模型族校准与提交门禁
│   ├── plan-v1.3.0.md        # P2 研究计划
│   └── project_report_v3.md  # 阶段报告
├── research/
│   ├── METHODS.md            # 方法原语库
│   └── literature_primer.md  # 文献路线整理
├── RESULTS.md                # 详细实验日志和负面结果库
└── README.md
```

原始数据、加工特征、模型权重、预测文件和提交文件均不进入 Git。比赛数据需从 Kaggle 获取，并在本地配置脚本使用的数据路径。

## 环境

- Python 3.11
- Polars / pandas / PyArrow
- LightGBM / XGBoost / CatBoost / scikit-learn
- PyTorch 2.x（本地 CUDA 环境与 Kaggle P100 环境需要分别处理）

部分脚本是实验快照，路径参数仍与本地数据布局绑定。复现实验前应先阅读 [docs/calibration.md](docs/calibration.md) 和对应脚本顶部的配置。

## 数据与安全

- 不提交 Kaggle 原始数据、派生 parquet、模型权重或 submission CSV。
- 凭证只通过环境变量或 Kaggle Secrets 注入，不写入代码和文档。
- 外部公开方案的预测仅作为 ensemble probe 管理，不进入本地分数校准模型。
