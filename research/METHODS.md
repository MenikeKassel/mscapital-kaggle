# MSCapital 金融预测方法库 (METHODS.md)

> 建立: 2026-08-12 | 理念: 方法考古 → 方法原语 → 组合创新 (GPT1 评审)
> 核心认知: **大部分高分方法不是第一次被发明** — 来源 = 前几年类似比赛 × 量化论文 × 通用 ML 新模型 × 比赛特化

## 0. 比赛定义 (数据形态分类)

> 匿名化高频市场事件数据 → 聚合成固定长度样本 → 预测未来收益 → cosine metric

搜索词(替代 "MSCapital high score"):
`Kaggle limit order book winning solution` / `market prediction 1st place` / `Optiver feature engineering` / `event based financial forecasting` / `order flow imbalance prediction` / `limit order book representation` / `financial tabular neural network`

## 1. 祖先比赛库 (方法迁移来源)

| 比赛 | 相似点 | 最值得偷什么 |
|---|---|---|
| **Optiver – Trading at the Close** (1st: hyd) | 股票/盘口/短周期 | imbalance, relative price, lag/diff, 时间结构, stock 聚合 |
| **Optiver Realized Volatility** (1st: Nearest Neighbors) | order book + trade event | 时间分桶, WAP, realized vol, near/far, 统计聚合 |
| **Jane Street Market Prediction** (1st: Supervised AE + MLP) | 匿名金融 tabular → future return | 先学特征表示再预测 (AE/MLP) |
| Jane Street Real-Time Market Data | 实时市场预测 | temporal validation, lag, cross-sectional |
| Ubiquant Market Prediction | 多资产匿名特征 | time_id, rank/relative, purged CV |
| G-Research Crypto Forecasting | 时序+多资产 | lag, return, rolling, 多资产相对变化 |
| JPX Tokyo Stock Exchange | 股票未来收益/ranking | cross-sectional normalization, rank |

## 2. 方法原语库 (A-F 六类)

### A. Level 特征 (当前状态)
- spread, bid/ask imbalance, depth, price distance, volume ratio
- 来源: 90 官方特征 / 我们的微观特征
- 实验: ✅ 已做 (v5)

### B. Change 特征 (刚才发生了什么)
- x_t - x_{t-1}, x_t/x_{t-1}, 多 lag diff, acceleration, reversal
- 来源: RealMLP 152 (diff/shift 7 gaps)
- 实验: ✅ 已做 (0726 特征)
- **新假设**: B001 diff 多尺度, B002 acceleration, B003 EWMA fast-slow 差

### C. Path 特征 (怎么走过来的)
- slope, max/min excursion, volatility, reversal 次数, sign persistence, cumulative change
- 来源: Optiver (realized vol), 0726 (价格趋势/自相关)
- **新假设**: C001 价格路径斜率, C002 excursion, C003 reversal 计数

### D. Event-time 特征 (事件速度)
- inter-arrival time, events/interval, burst intensity, mean/median interval, event acceleration
- 来源: LOB/Hawkes 文献, 0726 (时间间隔统计)
- **新假设**: D001 event rate, D002 price change/dt, D003 imbalance change/dt

### E. Regime/segment 特征 (前后段对比)
- near vs far, first 20% vs last 20%, recent vs old, 指数加权近期
- 来源: 0726 (near/far, 时间加权), P0 漂移发现
- 实验: ✅ 部分 (0726), E1 时间加权模型侧失败但**特征侧有效**

### F. Relative/cross-sectional 特征 (相对强弱)
- rank, zscore, market-relative return, sector-relative, cross-sectional deviation
- 来源: Ubiquant, JPX
- **新假设**: F001 跨样本 rank/zscore (MSCapital 每样本独立, 但可按月/时段聚合?) — 需评估适用性

## 3. 已吸收的方法 → 来源映射

| 方法 | 来源 | 状态 |
|---|---|---|
| 事件动力学特征 (152) | LOB 文献 × LLM 辅助 (DeepSeek) | ✅ 复刻, v7 兑现 +0.010 |
| RealMLP_RQ | arXiv 2407.04491 (通用 tabular) | ✅ 复刻, 融合 v7 |
| MultiStream (CNN+Transformer 三流) | DeepLOB × 标准 Transformer × 多模态 | 📦 lb142 包已下载, 训练需 grids (未公开) |
| 五路 ens5 ⊕ v10 融合 | 集成通用技巧 | ✅ 直接用其预测, v8 兑现 +0.007 |

## 4. 组合创新示例 (从已有方法生长)

- `price_diff_1 / mean_event_interval_5` = 单位事件时间内的价格变化速度
- `depth_imbalance_recent - depth_imbalance_far` = 盘口不平衡加速
- `EWMA(diff(price), 3) - EWMA(diff(price), 20)` = fast trend − slow trend
- 这些不需要等 Notebook: **已有方法 → 生长出二代特征**

## 5. 资源分配 (下一阶段)

```
60% 特征/表示迁移 (方法库 A-F → 新特征 → PSEUDO 验证)
20% 外部比赛方案考古 (Optiver/Jane Street top solutions)
10% validation (对抗验证 + PSEUDO 校准)
10% model/ensemble (融合权重, 多 seed)
```

## 6. 待办 (赛后/并行)

- [ ] 读 Optiver TATC 1st solution writeup (hyd)
- [ ] 读 Optiver RV 1st (Nearest Neighbors)
- [ ] 读 Jane Street 1st (Supervised AE + MLP)
- [ ] lb142 v10 模型代码深读 + grids 构建逆向 (grid_cache 未公开, 从 dataset.py/训练代码推断)
- [ ] 将每个新方案拆成方法原语入库
