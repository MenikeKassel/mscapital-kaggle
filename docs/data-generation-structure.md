# MSCapital 四文件数据生成结构取证 + 最像比赛 + 最优方案

> 日期: 2026-08-14
> 方法: 官方 Overview/Data 页实抓 + 本地 3000 样本采样取证 (脚本 `scripts/zz_forensics_datagen.py`)
> 官方定位 (Overview 原话): **"The actual market data is massive, so we've pulled out a simplified sample for everyone to play around with."** — 真实市场数据,简化抽样。

## 1. 四个文件的生成结构 (取证结论)

### 1.1 label.feather — 抽样窗口标签

| 事实 | 值 |
|---|---|
| 行数 | 1,257,637 (train), month 0-70 共 71 个月 |
| 每月样本数 | 17,187-17,852,**极其均匀** (min/max 差 <4%) → 每月固定预算随机抽样 |
| target | mean −1.2e-5, std 2.6e-3, 正占比 44.9%; 官方说明 = "future return (generation method not provided)" — **未来收益,生成方式刻意未公开** |
| 测试期 | month 71-108 (38 个月 OOD), 647,896 样本 |

生成方式: 从真实市场连续数据中,每月随机抽取 ~17.4k 个 10 分钟窗口作为样本 (预测时刻 = 窗口末端)。样本间无序列关系 (此前 P4 取证: 相邻 sample_id 边界跳变 ratio ≈ 1.0)。

### 1.2 market.feather — 600 秒等间隔 bar (核心表)

官方定义: **"Order book + aggregated trades, equally-spaced time bars"**,时间覆盖 ~600s (10 分钟)。

| 事实 | 值 |
|---|---|
| 每样本行数 | min 21 / p50 **189** / p99 201 / max 203 (~200 行/样本) |
| bar 间隔 | p50 **3.00s** / p90 3.12s / p99 11.9s / max 179.7s → **3 秒等间隔 bar 网格**,带少量缺失 (静默期), 时间戳 float32 抖动 |
| 列结构 | L1/L2 报价 (ask/bid price_1/2 + volume_1/2) + **bar 内成交聚合** (transaction_count/volume/avgprice) |
| 价格 0 | ask/bid price == 0 占比 0.1-0.4%,对应 volume 也全 0 → **空档哨兵** (该档无挂单) |
| avgprice NaN | 30.6% (该 bar 无成交) |
| spread | 有效报价行: p50 9.6e-4 (~0.1%), 无 spread=0、无交叉盘口 |
| 量纲 | 成交量整数, L1 深度 p50 ~7k、max 3.5 亿股 |

### 1.3 order.feather / transaction.feather — 60 秒原始事件流

官方定义: "Raw order flow" / "Raw trade flow",时间覆盖 **~60 秒** (预测前 1 分钟内)。

| 事实 | order | transaction |
|---|---|---|
| 每样本事件数 | p50 87 / p99 681 / max 983 | p50 49 / p99 449 / max 817 |
| 事件时间 | float32 连续 (真实事件时刻, 非网格) | 同左; 同秒批量成交 (同一时刻多次填充) |
| side | 0/1 ≈ 51.6/48.4 | ≈ 50/50 (主动买卖均衡) |
| action | 新单 75% / 撤单 25% | — |
| price 范围 | 0.0044..1.3063 (限价单可挂任意深度) | 0.9848..1.0110 (**成交只在 mid ±1.1%**) |
| volume | 97.9% 为 100 的整数倍, max 1,000,000 | 96.6% 为 100 的整数倍, max 974,200 |

### 1.4 关键一致性验证 (同一事件流,两种表达)

样本 738512 内: market bar 的成交聚合 vs transaction 表:
- 全 600s: bar count 总和 1720 vs tx 表仅 175 行 (tx 表只含 60s!)
- **≤60s 内: bar count 177 vs tx 表 175 行; bar volume 206,400 vs 202,900** (~1.7% 差, bar 边界归属误差量级) → **market bar 的 transaction_* 列 = 同一成交流的 3s 聚合; tx 表 = 最后 60s 的原始明细**
- market 快照时间 ∩ order 事件时间 = 0 (两套时间流不重合 → bar 网格独立于事件时刻)

### 1.5 匿名化指纹 (数据来自真实市场)

1. **价格按样本归一化到 ~1.0** (500 样本 mid 中位数 p50=1.0002; 打破真实 tick 网格 → 相邻唯一价差 = float32 ULP 量级 1.2e-7..6e-7, 呈现"连续"假象; 此前 P4 的"离散 tick"判断是 float32 中位数取整的误读)
2. **0.5 价簇**: 0.76% 样本 mid ≈ 0.5, 跨 71 个月+test 均匀 → 疑似第二标的/第二状态 (价格按不同基准归一化)
3. 无日期/标的代码/交易所字段, 只有 month; 时间戳加抖动
4. 量纲: 100 股手数 (97.9%) + 2-3% 零股 → 真实交易所 lot size 特征

### 1.6 生成结构总结 (一张图)

```
真实市场 LOB 连续流 (单一主标的, 价格匿名化到 1.0)
  → 按月切片 (109 个月: train 0-70, test 71-108)
  → 每月随机抽 ~17.4k 个 10 分钟窗口 (预测时刻 = 窗口末端 t=0)
  → 每窗口输出 4 层:
     label        = 窗口后的未来收益 (口径未公开)
     market       = 全 600s × 3s bar: L1/L2 报价 + bar 内成交聚合 (压缩层)
     order/tx     = 最后 60s 原始订单/成交事件流 (明细层, 10× 截断省体积)
```

**核心推论**: market.feather 是全历史信息的压缩层,order/tx 只是最后 60s 的细节放大。**600s 的 bar 序列 (3s 网格 ≈ 200 步, 建模 18 通道 = 11 raw + 7 derived; 原始表 13 列 = schema 列数, 非通道数) 是唯一尚未被自研管线完整建模的信息层** (152 特征只用 60s order/tx; 34 聚合特征无信号; 序列建模 P5-01/P5-02I 已实证: corr(y)=0.086 cosine, 信息在 ≤30s 短窗 + 跨通道同步 + 幅度状态)。

## 2. 与历史比赛对比: 哪个最像

| 排序 | 比赛 | 像在哪 | 不像在哪 | 冠军方法 (可迁移点) |
|---|---|---|---|---|
| **1** | **Optiver TATC 2023** | **10 分钟窗口 L1/L2 盘口快照 + 预测未来收益** — 结构最像 (market.feather 是其超集: 多了成交聚合 bar 和事件层) | 1s 网格 vs 3s; 横截面多股票 vs 单一匿名标的; MAE vs cosine; 无事件流 | 1st hyd: 盘口特征工程 (imbalance/depth/slope/wap) + GBDT 集成 + 后处理; **特征工程方向已被 152 特征覆盖** |
| **2** | **DRW Crypto 2025** | **测试期 = 未来月份 OOD + 匿名市场数据 + 方向型指标** (Pearson ≈ cosine) — 评估结构最像 | 分钟级匿名特征, 无 LOB 结构 | 1st: 3 层 MLP (SGD, 0.6MSE+0.4Pearson) + AE 合成特征 + **purged group TS CV 防泄漏**; 2nd: 线性模型+行组过滤 (数据质量>模型) |
| 3 | Optiver RV 2022 | 10 分钟窗口匿名特征, 窗口聚合特征工程 | 波动率目标非收益 | 1st: KNN 历史状态 + LGBM/1D-CNN/MLP 混合 |
| 4 | JS 2021 | 匿名表格 + NN 方法 (SAE latent) | 无市场结构 | 1st: Supervised AE + MLP, **CV split 内联合训练防泄漏** (单模型夺冠) |
| 5 | Ubiquant 2022 | 匿名特征 + 相关类指标 | 无市场结构 | 2nd: robust CV + LGBM |

**独特点 (Kaggle 上无先例)**: ①原始订单/成交事件流 (消息级 LOB, 只有学术界 LOBSTER/DeepLOB 有同构数据) ②600s+60s 不对称覆盖 (长历史 bar + 短窗口原始事件) ③cosine 指标 (方向型, 与 DRW 的 Pearson 最近) ④测试期 38 个月长 OOD。

主办方自己提示的方向 (Discussion): 特征 → Optiver RV/TATC; 模型 → 波形/传感器序列比赛 — 与上表一致: **TATC 是最像的"特征结构"参照, DRW 是最像的"评估结构"参照**。

## 3. 最优方案 (基于结构结论 + 最像比赛冠军方法)

### 现状
- 自研 v7/v8b = LB 0.135/0.142 (#30); 公开最强 LB142 = 0.142 (融合 0.6·ens5 + 0.4·v10); 前排 0.155+。
- 已吃干: 152 特征表格 (60s order/tx) × 树/NN/RealMLP、60s 事件流聚合、cosine loss 消融 (+0.001 实), 全部融合。
- **唯一未吃干: market 600s bar 序列** — P5-01 证据: MSE 臂无信号但 **cosine 臂 corr(y)=+0.086**, frozen 51-70 Δ+0.0009; LB142 的 market 网格 (200/400 步×12ch + cosine loss + online BN) 是唯一实证用它的人。

### 方案 (按性价比排序)

1. **P5-02: frozen market encoder → 32d latent → RealMLP** (主线, 已立项)
   3s bar 网格 200 步 × **18 通道** (11 raw + 7 derived: mid/spread1/depth1/imb1/spread2/depth2/imb2 — P5-01/P5-02I 实际口径), cosine loss 训练小 Conv1D (禁 Transformer), **latent 冻结**后与 152 特征拼接进 RealMLP — 防 TinyLOBERT corr 0.86-0.98 重演; 门禁: corr(latent, 152) < 0.70 且 frozen ΔPSEUDO ≥ +0.0015。⚠️ 历史文本中的 "13 通道 (11 raw + 2 派生)" 为旧规划残留, 实际实现一律 18 通道。
2. **cosine loss 全线回填** (已验证真实互补, 注意 4.14 配方纪律: 验证命题=提交命题)
3. **与 LB142 预测融合** (corr 0.82, 0.5/0.5 已实证 0.142; 新序列 latent 若 corr < 0.85 再进融合)
4. **DRW 式防泄漏纪律保持** (purged TS CV + 预训练类一律 inner-train 拟合 — JS2021 冠军教训)

### 结构上不值得再做 (证据)
- 60s 事件流再加工 (152 特征已榨干, 3 条表示路线全负)
- 34 聚合版 market 特征 (PSEUDO −0.0002, 序列必须建模不能聚合)
- 月漂移后处理 (P4-05 决定性负结果)
- target 口径逆向 (官方刻意不提供生成方式, 已试 5 路取证)

## 附: 脚本与复现

- 取证脚本: `scripts/zz_forensics_datagen.py` (3000 样本采样, ~3 分钟)
- 数据: `D:\mscapital-forecasting\data\raw\{train,test}\{market,order,transaction,label}.feather`
