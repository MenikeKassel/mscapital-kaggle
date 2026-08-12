# MSCapital 科研化方案 v1.3.0 (P2 阶段)

> 来源: GPT1 P2 方案 (2026-08-12) × Hermes 执行
> 状态: 已吸收; P2.0 云端执行中
> 前版: v1.2.0

## 核心路线 (GPT1)

> **外部方法考古 → 方法原语抽取 → MSCapital 特征化 → 低成本归因实验 → 强模型验证 → 相关性筛选 → 融合提交**

### 分层目标
| 阶段 | 目标 | 核心问题 |
|---|---|---|
| P2-A | 0.137~0.140 | 把 0.135 体系吃干榨净 (LB 已 0.142) |
| P2-B | 0.140~0.145 | 找到第二套有效信息表示 |
| P2-C | 0.145+ | 多信息源融合 / regime / 邻居方法 |

## P2.0 Attribution (第一优先, 云端已跑 = f0726 trees)

- A1: 152 特征 × LGBM ← **云端 RUNNING**
- A2: 152 特征 × CatBoost ← **云端 RUNNING**
- A3: 152 + R2 + 22micro × LGBM/Cat (待)
- A4: 152 + 官方 × RealMLP (待)
- **回答: 0.134 来自特征还是架构?**

## P2.1 Dynamics V2 (40% 资源)
- 多尺度 diff (1/2/4/8), acceleration, fast-slow (EWMA 3/8/16), path (slope/reversal/excursion)
- 每类单独 ablation, 不一次塞 200 特征

## P2.2 Event Process (重点)
- event rate, change/dt, recent/old intensity, price/depth/imbalance velocity

## P2.3 Dynamic Microstructure
- dynamic imbalance/spread, price×liquidity dynamics, 多尺度 interaction

## P2.4 Relative Dynamics
- 只 rank 动态特征 (rank(diff), rank(velocity), rank(imbalance_change)) — 不做全量 relative

## P2.5 Market State KNN (替代 alpha)
- 状态向量 (vol, imb, spread, event intensity, price dynamics) → K 近邻 → weighted mean(target)
- 期望 corr(RealMLP, KNN) = 0.5-0.7 → 高融合价值

## 提交门禁 (制度化, v6 教训)

| Gate | 检查 | 失败处理 |
|---|---|---|
| 1 | PSEUDO/temporal folds 稳定 | 不提交 |
| 2 | test pred 分布 (mean/std/min/max/分位数) | 异常不提交 |
| 3 | std_test/std_valid 尺度比 | 严重偏离禁止 |
| 4 | corr vs v5/RealMLP/最佳提交 | 记录 |
| 5 | 融合 sanity (std 不突变, mean 不漂移) | 禁止 |

## 资源分配
40% Dynamics V2/Event | 20% 方法挖掘 | 15% micro×dynamics | 10% tree attribution | 10% ensemble | 5% 调参

## 模型侧冻结
Transformer/TCN/新架构冻结; 只留 RealMLP(多seed) + GBDT(LGBM/CAT) + KNN

## 执行中 (云端)
- f0726 trees (P2.0 A1/A2) RUNNING
- realmlp_pseudo (v7 本地分校准) 待 dataset 就绪
