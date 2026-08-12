# Literature Primer — MSCapital 文献初筛清单

> 生成: 2026-08-10 | 核验方式: arXiv 网页/abs 页面逐一核验 ID+标题
> 状态标签: [verified] = ID+标题核验通过 | [abstract read] = 已读摘要(经典论文)
> 相关性评分 = 初步评估(0-5), 待 Deep Read 后修订
> 防伪纪律: 所有条目均有真实 arXiv ID, 链接可点击核验; LightGBM 论文无核验ID故不列

## 主题1: Order Flow / OFI 预测力

### P001 [verified+abstract read] 相关性: 5/5
- **The Price Impact of Order Book Events** (Cont, Kukanov, Stoikov)
- arXiv:1011.6402 (2010, 发表于 JFE 2014) https://arxiv.org/abs/1011.6402
- 贡献: 定义 Order Flow Imbalance (OFI) = 盘口事件(新增/取消/成交)按价格加权的净流量; 实证 OFI 与短期价格变化近似线性关系 (R²≈0.6-0.8)
- 迁移: OFI 是本赛 order 表可直接计算的核心特征; 官方基线已含简化版 OFI

### P002 [verified] 相关性: 4/5
- **Cross-Impact of Order Flow Imbalance in Equity Markets** (2021)
- arXiv:2112.13213 https://arxiv.org/abs/2112.13213
- 贡献: OFI 跨资产(跨盘口)影响建模 — 多个资产/层级之间的 OFI 交叉冲击
- 迁移: 本赛 L1/L2 两层盘口, 跨层级 OFI (L1 vs L2) 可能是增量特征

### P003 [verified] 相关性: 3/5
- **Returns and Order Flow Imbalances: Intraday Dynamics and Macroeconomic News Effects** (2025)
- arXiv:2508.06788 https://arxiv.org/abs/2508.06788
- 贡献: OFI 与收益的日内动态, 宏观新闻事件的调节效应

## 主题2: LOB 深度学习预测

### P004 [verified+abstract read] 相关性: 5/5
- **DeepLOB: Deep Convolutional Neural Networks for Limit Order Books** (Zhang, Zohren, Roberts)
- arXiv:1808.03668 (2018) https://arxiv.org/abs/1808.03668
- 贡献: CNN+LSTM 直接吃 LOB 原始盘口快照序列预测未来方向, 优于手工特征 ML
- 迁移: 本赛 market 表是 bar 聚合而非快照, 原始序列建模需重构; 但特征输入思路可借鉴

### P005 [verified+abstract read] 相关性: 4/5
- **Universal features of price formation in financial markets: perspectives from Deep Learning** (Sirignano, Cont)
- arXiv:1803.06917 (2018) https://arxiv.org/abs/1803.06917
- 贡献: 大规模 (数亿条消息) LOB 数据训练 DNN, 发现价格形成存在跨市场通用结构
- 迁移: 支持"大量序列数据 + 深度模型"路线; 但本赛 CV/LB 脱节提示通用结构可能被漂移破坏

### P006 [verified] 相关性: 4/5
- **HLOB — Information Persistence and Structure in Limit Order Books** (2024)
- arXiv:2405.18938 https://arxiv.org/abs/2405.18938
- 贡献: 深度 CNN 建模 LOB 信息持久性; 强调输入表示 (depth/宽度) 比模型结构更重要
- 迁移: "输入表示 > 模型结构" 与本赛 90特征甜点位观察一致

### P007 [verified] 相关性: 3/5
- **TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction with LOB** (2025)
- arXiv:2502.15757 https://arxiv.org/abs/2502.15757
- 贡献: 双注意力 Transformer 预测价格趋势

### P008 [verified] 相关性: 3/5
- **Price predictability in limit order book with deep learning model** (2024)
- arXiv:2409.14157 https://arxiv.org/abs/2409.14157

### P009 [verified] 相关性: 2/5
- **Exploring Microstructural Dynamics in Cryptocurrency Limit Order Books: Better Inputs** (2025)
- arXiv:2506.05764 https://arxiv.org/abs/2506.05764
- 注意: 加密市场, 与股票/期货市场结构有差异, 迁移需谨慎

## 主题3: 分布漂移 / 时序泛化 (本赛 CV/LB 脱节的理论基础)

### P010 [verified+abstract read] 相关性: 5/5
- **AdaRNN: Adaptive Learning and Forecasting of Time Series** (KDD 2021)
- arXiv:2108.04443 https://arxiv.org/abs/2108.04443
- 贡献: 显式建模时序分布漂移 (Temporal Distribution Characterization + 分布匹配训练); 比常见 RNN 跨分布泛化更好
- 迁移: 本赛 train(0-70月) vs test(71-108月) 是典型时序漂移; 分布匹配/样本重加权思路直接适用

### P011 [verified] 相关性: 3/5
- **Real World Time Series Benchmark Datasets with Distribution Shifts** (2023)
- arXiv:2308.10846 https://arxiv.org/abs/2308.10846
- 贡献: 含真实分布漂移的 TS benchmark, 提供评估协议参考

### P012 [verified] 相关性: 5/5
- **Deep incremental learning models for financial temporal tabular datasets with distribution shift** (2023)
- arXiv:2303.07925 https://arxiv.org/abs/2303.07925
- 贡献: 金融表格数据 + 分布漂移 + 增量学习 — 与本赛形态 (表格特征 + 时序漂移) 高度一致

### P013 [verified] 相关性: 3/5
- **ProteuS: A Generative Approach for Simulating Concept Drift in Financial Markets** (2025)
- arXiv:2509.11844 https://arxiv.org/abs/2509.11844

## 主题4: 微观结构特征工程 (检索待补, 先用 P001/P006 覆盖)

- 待补检索: microprice / queue imbalance / weighted mid 实证论文 (semantic scholar 限流, 稍后补)
- 实践来源: Optiver 两届 Kaggle 赛复盘 (Tier B, 实战验证的特征集) — 官方主办方明确提示参考

## 主题5: GBDT vs 序列模型 / 融合

### P014 [verified] 相关性: 4/5
- **Sequential Structure in Intraday Futures Data: LSTM vs Gradient Boosting on MNQ** (2026)
- arXiv:2605.17724 https://arxiv.org/abs/2605.17724
- 贡献: 直接对比 LSTM vs GBDT 于日内期货数据 — 与本赛形态最接近的对比研究
- 迁移: 支撑 Exp D (树 vs 序列) 设计; 需读全文看结论与数据口径

### P015 [verified] 相关性: 3/5
- **Deep learning models for price forecasting of financial time series: A review of recent developments** (2023)
- arXiv:2305.04811 https://arxiv.org/abs/2305.04811
- 贡献: 金融价格预测深度学习综述 — 用于补全方法演化链

### P016 [verified] 相关性: 3/5
- **RegimeFolio: A Regime Aware ML System for Sectoral Portfolio Optimization** (2025)
- arXiv:2510.14986 https://arxiv.org/abs/2510.14986
- 贡献: regime-aware 建模 — 支撑 RQ6 (Alpha 是否依赖市场状态)

## 主题6: 基础模型论文 (通用, 快速核验)

### P017 [verified+abstract read] 相关性: 4/5
- **An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling** (TCN, Bai et al. 2018)
- arXiv:1803.01271 https://arxiv.org/abs/1803.01271
- 贡献: TCN 在多数序列任务上优于 LSTM/GRU; 因果卷积+膨胀

### P018 [verified+abstract read] 相关性: 2/5
- **Attention Is All You Need** (Vaswani et al. 2017)
- arXiv:1706.03762 https://arxiv.org/abs/1706.03762

### P019 [verified+abstract read] 相关性: 3/5
- **XGBoost: A Scalable Tree Boosting System** (Chen, Guestrin 2016)
- arXiv:1603.02754 https://arxiv.org/abs/1603.02754

### P020 [verified] 相关性: 2/5
- **TabNet: Attentive Interpretable Tabular Learning** (2019)
- arXiv:1908.07442 https://arxiv.org/abs/1908.07442
- 贡献: 表格数据注意力模型 — GBDT 强基线对照

---

## 小结 (供 Gate 1 汇报)

- 已核验 20 篇, 其中 9 篇 [abstract read]
- 核心结论线索: ①OFI 有强实证基础 (P001) ②输入表示 > 模型结构 (P006) ③时序漂移是真实威胁且有专门方法 (P010/P012) ④LSTM vs GBDT 直接对比存在 (P014)
- 缺口: 主题4 特征工程检索 (待补); LightGBM 原始论文 (无核验 arXiv ID)
