# 类似比赛侦察与 SOTA 方法 (2026-08-13)

> 目的: 找与 MSCapital 数据/任务形态最接近的比赛, 提取已验证的顶级方法, 评估对 MSC 的可迁移性。
> 方法: Kaggle 实页抓取 (DRW/Hull writeups) + arXiv 检索 + 项目内已有溯源 (method-provenance.md)。

## 一、相似度排序

| 比赛 | 时间/规模 | 数据形态 | 评价 | 与 MSC 的相似点 | 差异 |
|---|---|---|---|---|---|
| **DRW - Crypto Market Prediction** | 2025.5-7, $25K, 1,091 队 | 分钟级 bid/ask qty + 交易量 + **780 匿名特征** | Pearson | ①未来数据评分 (私有榜用更新数据, 公开榜不评分) → **时间漂移即赛题**; ②价格方向回归; ③匿名特征可做聚类/选择 | 分钟级+匿名, 无逐事件 LOB, 无 60s 窗口 |
| **Hull Tactical - Market Prediction** | 2025.9-2026.6, $100K, 3,677 队 | 日频 S&P500, 197 列 M/E/I/P/V/S/MOM/D 分组特征 | custom (收益/波动导向) | 两阶段: 公开测试=训练集尾部 (公开 LB 无意义), **最终用截止后未来数据评分**; 漂移是核心 | 日频宏观, 非微观结构 |
| Optiver - Trading at the Close | 2023, 1,422 队 | 10 分钟快照, 股票×时间横截面 | RMSPE | 盘口 imbalance/深度特征族 (已吸收进 M05) | 横截面+分钟级 |
| Optiver Realized Volatility | 2022 | 秒级匿名特征 | RMSPE | KNN 冠军 (M04 已吸收) | 波动率回归非方向 |
| Jane Street Real-Time Market Data | 2024 | 匿名特征, 实时评分 | utility | adaptive+static 思路 (已标内部假设) | 二分类+权重 |
| Ubiquant Market Prediction | 2022 | 匿名特征 | Pearson | robust CV (已吸收) | 分钟级匿名 |
| JPX Tokyo Stock Exchange | 2022 | 匿名特征 | Pearson | 漂移处理 | 匿名, 无原始事件 |

**共同规律: 近 4 年的金融回归比赛全在打"时间漂移"这一仗** (DRW/Hull 甚至直接设计成未来数据评分)。MSC 的月度漂移不是孤例, 是此类比赛的共同核心难题。

## 二、最先进方法 (冠军级, 已逐条抓取原文)

### DRW 1st (A_A) — MLP 主角 + 特征工程
- 模型: 3 层 MLP 为主 (单模 public 0.124/private 0.131), XGB 为辅
- CV: purged group time series split, 6 组 (约 2 个月/组), gap=1
- 特征选择 (作者称最关键): ①聚类 (corr>0.6 归簇) 取 medoid → 780→60; ②删与 target corr≤1e-4 → ~40; ③XGB 6 折 SHAP top20 逐折求交集再并集 → ~30; ④线性组合合成特征; ⑤**AutoEncoder 合成 8 个 deep features (作者明言很重要, CV/LB 双提升)**
- 训练技巧: **SGD 而非 Adam/AdamW** (实测后两者差); **loss = 0.6×MSE + 0.4×Pearson, 验证用 Pearson** (纯 MSE 训 NN 相关性差, 但 XGB 用 MSE 没事)
- 观察: 数据更新后旧模型失效 = 特征随时间失去预测力 → "回收"曾被丢弃的特征

### DRW 2nd / 25th (EL Younes) — 数据质量 > 模型复杂度
- 同一个线性模型 + 同一组特征, 只改数据准备: 2nd (分钟级时间特征 Y_M_D_H_M) vs 25th (小时级 Y_M_D_H)
- 方法: 时间分组 → 特征聚类找相似行组 → 迭代删除"坏"行组 (验证变好就删)
- 结论: 干净训练数据可击败复杂模型 (与本项目 N002 教训同构)

### Hull 4th (YannFb) — 无 ML, 组合工程 (前 3 名未发方案)
- 无模型无调参: 规则化短期反转 alpha (赛前已独立验证) + 逆波动率加权 + 波动率目标 overlay + 杠杆 clip
- 洞察: 指标奖励"低波动市场收益" → **组合构建 > 预测精度**; 特征不预测收益也能降噪
- 对 MSC 迁移有限 (MSC 无组合环节), 但"少调参防过拟合"与项目纪律一致

## 三、arXiv 最新 LOB 预测 (2025-2026, 监控清单)

| 论文 | 日期 | 一句话 | 优先级 |
|---|---|---|---|
| In-Network Market Prediction Using ML and LOBs (2608.02424) | 2026-08 | 网络内 LOB 预测架构 | 低 (偏系统) |
| Latency-Efficient Architecture for LOB Prediction (2606.25986) | 2026-06 | 推理-计算前沿分析 | 低 (工程向) |
| Early Detection of Latent Microstructure Regimes (2604.20949) | 2026-04 | 微观结构 regime 检测 → 可与 M06 regime 结合 | 中 |
| KANFormer for Fill Probabilities (2512.05734) | 2025-12 | KAN 预测成交概率 | 低 (分类向) |
| Detecting Multilevel Manipulation via Cascaded Contrastive Repr (2508.17086) | 2025-08 | 对比表示检测操纵 | 低 |
| ByteGen: Tokenizer-Free Generative Model for Orderbook Events (2508.02247) | 2025-08 | 生成式订单流模型 | 低 (生成向) |

## 四、对 MSC 的可迁移清单 (按成本×预期排序)

1. **AE 合成特征** (DRW 1st 验证): 现有 152+22+M01-M05 特征 → 浅层 AE → 8-16 个 deep features 加入融合。
   与 SimLOB 路线同思想, 但成本极低 (MLP AE), 是"表示压缩"的快速试验。
2. **混合 loss: 0.6×MSE + 0.4×cosine** (DRW 用 Pearson, 我们指标即 cosine): RealMLP/Cat 训练直接换 loss,
   验证仍用 cosine。低风险, 一票实验。
3. **MLP 优化器: SGD vs Adam** (DRW 1st 实测): 检查现有 RealMLP 复刻优化器, 若为 Adam 可对比 SGD。
4. **特征去冗余: 聚类 medoid** (DRW 1st): M01-M05 累计特征已数百, 聚类取代表特征防维度膨胀, 呼应 90 特征甜点位。
5. **行级/时段过滤** (DRW 2nd): 用 canonical OOF 找"毒月份/毒时段"剔除, 验证是否改善。呼应 N002。
6. **Hull 4th 思想**: "特征不预测收益也能降噪" — 评估特征时除 alpha 轴外保留稳定化价值维度。

## 五、结论

- 冠军方法再次验证项目既有纪律: **表示与数据质量 > 架构** (DRW 1st 特征工程、DRW 2nd 数据过滤、Hull 4th 无模型)。
- 与 MSC 最相似的 DRW 1st 提供了 3 个未试过的低成本杠杆: **AE 合成特征、MSE+cosine 混合 loss、SGD 训练**。
- 漂移是同类比赛共性, MSC 的 canonical OOF / PSEUDO 门禁制度是正确武器, 继续沿用。
