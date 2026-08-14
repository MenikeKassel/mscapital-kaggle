# P4-01(a) Market 600s Information Forensics Report

> 日期: 2026-08-14 | 阶段: P4-01(a) 信息存在性取证 (test split)
> 数据: test/market.feather (118.36M 行, 647,896 样本) + d_analysis (|d| = |unit(ref)−unit(v7)|) + f0726 152 特征
> 门禁: Gate A 分布证据 / Gate B 增量可预测性 / Gate C regime 集中

---

# 1. Executive Conclusion

## ✅ CONDITIONAL GO — market.feather 值得进入正式建模（序列级）

**三条门禁全部通过**，且增量在剔除活动度/波动代理后依然成立：

| 门禁 | 结果 | 判定 |
|---|---|---|
| Gate A 分布证据 | 47/50 描述符跨折方向一致(≥0.90)，高\|d\| 样本路径特征显著不同 | ✅ 通过 |
| Gate B 增量可预测性 | C(152+market) vs A(152)：top10-AUC +0.019，5 seeds 全部为正 | ✅ 通过 |
| Gate C regime 集中 | n_snap 效应在高活动×高波动格最强 (d=+1.56)，E[\|d\|] 随波动率单调 | ✅ 通过（部分） |

**但注意**：market 单独 (B) 弱于 152 (A)（AUC 0.786 vs 0.822）——market 不是独立主导源，是 **152 之上的增量层**。且所有描述符是聚合级；序列级信息存在性仍需 (b) 验证。

**H4 prior 45% → 更新后 55%**（研究优先级概率，非正式后验）。

---

# 2. Data Integrity

| 项 | 结果 |
|---|---|
| schema | 13 列: sample_id, seconds_before_predict + 11 盘口字段 (L1/L2 bid/ask price+vol, tx avgprice/volume/count) |
| 总行数 | 118,359,166 (test) |
| 样本数 | 647,896 (与 d_analysis 100% 对齐) |
| NaN | 仅 transaction_avgprice: 26.41% (31.26M)；其余 0 |
| 样本长度 | min=18, p10=148, median=194, p90=199, max=200 |
| 时间覆盖 | seconds_before_predict: 0 ~ 597.2s (≈600s) |
| 重复 | (sample_id, seconds) 唯一对 100%，无重复 |
| 备注 | **max=200 恰好等于 LB142 market_len=200 上限** → LB142 可能直接截断到 200 步 |

⚠️ 教训沿用: transaction_avgprice 26% NaN → 所有 avgprice 派生量必须 NaN-safe（本报告全部用 fill_null/skip-null 处理）。

---

# 3. High-|d| vs Low-|d|（Gate A 证据）

Top/Bottom 10% |d|（n=64,789），Cohen's d，fold 方向一致性（10 随机折）：

| market 统计量 | high-|d| 均值 | low-|d| 均值 | Cohen's d | fold 一致 |
|---|---:|---:|---:|---:|
| n_snap | 193.10 | 177.79 | **+0.72** | 1.00 |
| imb_std | 0.281 | 0.256 | +0.36 | 1.00 |
| mid_range | 0.0111 | 0.0036 | +0.28 | 1.00 |
| mid_std | 0.0033 | 0.0010 | +0.22 | 1.00 |
| spread_widen_freq | 0.286 | 0.230 | +0.21 | 1.00 |
| mid_seg_std_early | 0.0017 | 0.0005 | +0.16 | 1.00 |
| mid_vol_rel | 0.0038 | 0.0011 | +0.16 | 1.00 |
| imb_mean | 0.510 | 0.526 | −0.14 | 1.00 |
| depth_std | 176,198 | 75,034 | +0.10 | 1.00 |

47/50 特征 fold 方向一致率 ≥0.90 → **分布差异跨折稳定，非单月孤例**（test 无 month 字段，用随机折代替，已注明限制）。

**解读**：高|d| 样本 = 快照更密、波动更大、深度不平衡波动更大、价差扩大更频繁。与 P4-04 画像（高活动×高波动）完全吻合。

---

# 4. Temporal Path Evidence

分段（early/mid/recent tercile）对比:

- **volatility path**：mid_seg_std 三分段全显著 (d≈+0.15~0.19)，early/mid/recent 无单调差异 → 波动水平差异是全程的，非尾部特定。
- **evolution ratios 全部 ~0**：
  - vol_recent_early d=+0.04 (5%) / spread_recent_early d≈0.00 / depth_recent_early d≈0.01 / mid_trend d≈−0.02 / spread_trend d≈0.00
- **结论**：600s 内"如何演化"（early→recent 变化方向）**不区分**分歧；区分力在**水平/波动幅度/密度**，不在路径形状。这反驳了"演化特征有增量"的直觉，但支持"快照密度与波动水平"作为信息载体（序列级 12 通道可能仍在步级时序里携带水平类信息）。

---

# 5. Conditional Regime Evidence（Gate C 部分）

E[|d|]（×1e4）按 quintile（rank-based）：

| 维度 | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---:|---:|---:|---:|---:|
| activity (o_sec_new_count) | 5.15 | 4.67 | 4.68 | 4.68 | 4.66 |
| **volatility (t_price_volatility)** | **4.20** | 4.52 | 4.78 | 5.14 | **5.21** |

- |d| 随波动率**单调递增**（Q1→Q5: 4.20→5.21）✅ 与 P4-04 一致
- 活动度五分组近似平坦（Q1 略高）——分歧主要由**波动率**驱动，非活动度本身

per-cell Cohen's d（关键特征，行=activity Q, 列=vol Q）:

**n_snap**（最强）:
```
  +0.15  +0.20  +0.30  +0.41  +1.56   ← 高活动×高波动 cell 效应最大
  +0.26  +0.24  +0.32  +0.43  +1.00
  +0.22  +0.30  +0.35  +0.42  +0.72
  +0.18  +0.28  +0.36  +0.33  +0.49
  +0.16  +0.20  +0.25  +0.28  +0.26
```

**mid_range / mid_std**：所有格子正 (d≈0.2~0.7)，高活动×低波动最强 → 水平波动全 regime 有效。

**imb_std**：所有格子正 (0.2~0.8)，低活动更强。

**Gate C 判定**：n_snap 的效应在高活动×高波动格暴增 (+1.56) → **部分通过**；但多数特征全 regime 有效，非严格集中。

---

# 6. Disagreement Predictability（Gate B 证据）

Ridge OOF 5-fold，预测目标 |d|：

| 模型 | OOF R² | OOF MAE | top10-AUC |
|---|---:|---:|---:|
| A: 152 features | −0.481 | 0.000508 | 0.8224 |
| B: 50 market desc | −0.524 | 0.000503 | 0.7857 |
| **C: 152 + market** | **−0.444** | 0.000510 | **0.8416** |

**增量 (C−A)**：dR2 = +0.037，dAUC = **+0.019**，5 seeds 全部为正（min +0.0192）。

**敏感性**（剔除 7 个活动/波动代理列 n_snap/mid_std/mid_range/mid_vol_rel/seg_stds 后）：
- C−A dR2 = +0.035，dAUC = **+0.019** → **增量不是活动度代理的重复**，来自路径描述符本体。

R² 为负 = |d| 本身噪声主导（模型间差异大部分不可预测），但**排序能力 (AUC) 是真实的**：market 描述符让"哪些样本会分歧"的排序提升 0.019。

---

# 7. H4 Update

```
H4 prior:                                  45%
Evidence after P4-01(a):
  FOR:
    - Gate A: 47/50 描述符稳定区分 high/low |d|（fold 一致 ≥0.90）
    - Gate B: 152+market 增量 AUC +0.019，去代理后仍 +0.019
    - Gate C: n_snap 效应在高活动×高波动格 +1.56（机制闭环，对应 LB142 网格输入的 200 步截断）
    - E[|d|] 随波动率单调 → 分歧样本 = LB142 掌握更多信息的样本（与 P4-04 一致）
  AGAINST:
    - market 单独 (B) 弱于 152 (A) → 不是独立主导源
    - 演化类特征 (recent/early ratios) 零区分 → 信息可能只是"快照密度/水平"而非复杂路径形状
    - 聚合描述符的增量在 target 上未必变现（P4-01(b) 才能回答）
Updated working probability:               55%
```

---

# 8. Next Decision

## ✅ A — RUN P4-01(b): market-only donor（正式 target 实验）

理由：
1. 三条门禁全过，增量真实（去代理稳健）
2. 与 LB142 源码的 200×12 网格输入直接对应（n_snap max=200 = 网格长度）
3. 下一题是"序列级 market 表示能否变现 target 增量"——只有 (b) 能回答

**P4-01(b) 设计（按 GPT 规格）**：
- market-only 轻模型：200 步 × 11 通道 → Conv1D/TCN 小网络（或先做 200×11 的 light GBM on flattened patches）
- 输出三数：PSEUDO/OOF、corr(pred_market, pred_M01A)、corr(pred_market, residual_M01A)
- 门禁：ΔPSEUDO ≥ +0.0015（强）或 corr(pred_market, M01A)<0.80 且残差正相关（弱 donor）
- 失败则 STOP market line，返回 P4-05

**执行前置**：canonical_residual_oof.npz 缺失 → 需先重建 canonical OOF（R21_30/R31_40/R41_50/R51_60/R61_70 五块已在 output/canonical_oof_blocks/，缺合并）或现场重建残差。

---

# 9. 研究原则复核

```
新信息源 > 信息存在性证明 > 目标相关性 > 表示方式 > 模型架构 > 超参数
        ✅ (已完成)          → (b) 回答     → (b) 决定     → (c) 再谈
```
