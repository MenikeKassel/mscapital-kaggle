# MSCapital 项目阶段报告 v3 (2026-08-12 13:15)

> 前版: v1 (00:55) / v2 (02:50) | 本轮: RealMLP 复刻突破 (0.125 → 0.135), 特征源战略转向

## 1. 比赛与当前成绩

- **比赛**: MSCapital – Real Financial Market Forecasting (Kaggle 社区赛, cos 指标, 个人赛)
- **当前**: LB **0.135**, 排名 **#48 / 107 队** (截止 2026-08-12 02:23)
- **距离**: 前 10 需 ~0.151 (差 0.016); 榜首 0.155
- 截止: 2026-10-09 (约 2 个月)

## 2. 提交历史 (8 次)

| 版本 | 内容 | LB | 校准 |
|---|---|---|---|
| v1 | LGBM+XGB+MLP (CV1 权重) | 0.122 | CV1 0.139 |
| v2 | +CatBoost | 0.122 | 同 |
| v3 | temporal 权重重估 | 0.122 | PSEUDO 0.130 |
| v4 | **R2 归一化特征** | **0.123** | PSEUDO 0.131 |
| v5 | R2+22 微观特征全量 | **0.125** | PSEUDO 0.135 |
| v6 | +TCN 全量 (w=0.07) | **0.082** ❌ | PSEUDO 融合 0.138 |
| v7 | v5+RealMLP 复刻 (0.8/0.2) | **0.135** 🚀 | corr 0.83 |
| v7b | RealMLP 0.25 | 0.134 | 权重 0.20 最优 |

**校准规律**: 表格 PSEUDO-LB38 ≈ LB+0.010; 序列模型 PSEUDO 完全不可信 (v6 教训)。

## 3. 实验体系 (30+ 实验, 全记录 RESULTS.md)

### P0 验证体系 (双 GPT 评审驱动)
- Adversarial Validation: 三组 AUC 0.73-0.78, 价差/深度/波动=漂移主力
- Temporal Matrix (7 folds × 5 模型): CV1 模型选择能力证伪 (MLP 假冠军, 后经 P0.5-B 修正为评估协议 bug)
- P0.5-C 漂移干预: **归一化 (R2) 4/4 folds 全正 (+0.0023)**, 远期提升最大 → v4/v5 兑现

### P1 特征与表示
- 22 个微观结构特征 (P1-1a): 融合 PSEUDO +0.0036 (v5 兑现 LB +0.002)
- 相对化第二轮 (P1-1e): 负面 (N006)
- **RealMLP 复刻 (P1-1f/g): 152 个事件动力学特征 + RealMLP_RQ → v7 LB 0.135 (+0.010!)** — 项目最大突破
- TCN 双塔 (P1-2): 3-fold 融合全正, 但 test 分布外退化 (v6 灾难) → 序列模型弃用

### 负面结果库 (N001-N006)
时间加权、cos loss、>90特征、Transformer、TCN 融合、特征相对化第二轮

## 4. 核心战略领悟

> **换特征源 (信息表示) > 换模型 (架构/调参)**

- 官方 90 特征空间天花板 ≈ 0.125 (v1-v5 用尽所有模型/融合/归一化手段验证)
- RealMLP 152 事件动力学特征 (diff/shift 动态、时间指数加权、事件顺序、间隔统计、near/far 段对比) 单模型 0.134
- 融合 v7 = 0.135 (corr 0.83, 真实互补)
- 文献印证: HLOB "输入表示 > 模型结构"; GPT 评审早有提示

## 5. 当前进行中 (云端)

- **f0726 树模型 kernel** (kasselmenike/msc-f0726-trees v4): 0726 特征 × CatBoost/LGBM, PSEUDO 诚实验证 + 80 万作者协议 + 全量 test 预测
- 独立 dataset (msc-f0726-data, 833MB) 已挂载 (修复此前 dataset 快照问题)
- 预计 40-60 分钟完成

## 6. 下一步计划

```
① 云端 f0726 树模型结果 → 三路融合 (v5表格 + RealMLP + 0726树模型)
   → v8 提交 (预期 0.136-0.138)
② RealMLP 多 seed 集成 (更强组件)
③ 0726 特征 × 官方特征融合 (双特征源树模型)
④ 配额管理: 云端 GPU 剩 ~13h, 提交 5次/天 (今天已用 2, 剩 3)
```

## 7. 资产清单

```
代码: D:\mscapital-kaggle\ (scripts 00-42 + RESULTS.md + docs/plan-v1.2.0)
数据: D:\mscapital-forecasting\ (raw + processed + f0726 特征 + p12_out)
参考: reference/ (lgb_baseline, realmlp, salute_rib, transformer, rfmf_0726 源码)
云端: msc-f0726-trees (kernel), msc-f0726-data + msc-0726-featbuild (datasets)
环境: 本地 4060Ti 8GB (轻载); Kaggle GPU ~13h 配额
```

## 8. 关键经验 (给后续评审)

1. 科研流程有效: 对抗验证 → 干预验证 → PSEUDO 校准 → LB 兑现 (v4/v5/v7 全部命中)
2. 但流程也有盲区: 序列模型 PSEUDO 验证失效 (v6), 教训 = test 上 corr 检查是最后防线
3. 用户驱动价值: "为什么不参考 RealMLP" 直接带来最大突破 — 外部方案侦察应更主动
4. 云端工作流已打通: dataset 版本快照坑 (独立 dataset 解决), P100 torch 兼容 (2.2.2+numpy<2)
