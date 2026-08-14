# P6R Preregistration — Retrieval-Conditioned Residual Alpha

> 日期: 2026-08-14 | 状态: **执行前定稿**（P6R-00 立即执行；P6R-01 条件触发）
> 前置: `docs/p6r_repo_audit.md`（数据契约 A–H）
> 纪律: 本文件在查看任何 frozen outer 结果前定稿。修订必须 changelog + 理由。

## 1. 研究问题与假设（来自任务书）

- H1: 相似历史 Market State 的 baseline residual 可预测当前 residual（局部可预测性）
- H2: 同一 event signal 的 beta 随 Market State 变化（Conditional Alpha / varying coefficient）
- H3: OOD support gate 能识别"不应修正"的样本并避免爆炸
- 最终目标: 回答 Q1–Q11（见 `docs/p6r_experiment_report.md` 模板），不是造新 y predictor

## 2. 数据契约（全部来自 audit，不可修改）

| 项 | 值 |
|---|---|
| 残差目标 | canonical OOF rows（`output/canonical_residual_oof/canonical_residual_oof.npz`，month 21–70，885,936 行） |
| 残差定义 | r = y − β·y0；β = dot(y0,y)/dot(y0,y0)，每 outer 从可见 view（`outer_residual`，months ≤ visible_oof_end）拟合；β=1 变体作诊断 |
| 邻居库 | canonical OOF rows；**严格断言 month_neighbor < month_query** |
| 状态表示 z | E02 11 context 特征（`revol_lite_train.parquet`），按 `context_feature_names()` 顺序 |
| 事件特征 X (P6R-01) | f0726_train.parquet 152 特征中预注册 40 个核心特征（见 §6） |
| α 选择区（tune） | RESIDUAL_INNER_SPLITS: PSEUDO 27–32 / H2 31–40 / T3,T4 41–50 |
| α 网格 | {0.05, 0.10, 0.15, 0.20, 0.30}（E01 惯例: y_hat = RMS(y0) + α·RMS(r_hat)） |
| 冻结评估区 | NESTED_SPLITS outer_valid: PSEUDO 33–70 / H2 51–60 / T3 51–60 / T4 61–70 |
| 主锚基线 | c4-frozen 生产基线（`output/c4_protocol_closed_final/...`，与 E01/E02 可比） |
| 辅锚基线 | canonical OOF 基线（同月份行） |
| scaler | 只从各 outer 的 inner-train 月拟合（PSEUDO 21–26 / H2 21–30 / T3,T4 21–40），应用于 bank+queries |

**银行设计（两个注册变体，主 = V1）**:
- V1 扩展银行（主）: 查询月 m 的银行 = canonical rows month < m（无上限；与任务书"只允许搜索该 query 月份之前的历史 OOF 样本"一致，E02 同款）
- V2 封顶银行（变体）: 银行 = month < m 且 ≤ visible_oof_end(outer)（E01 模型训练范围同款）
- 两变体都满足 month_source < m（测试强制）

## 3. P6R-00 — Retrieval Residual Mean（立即执行）

**候选（预注册，共 8 个 = 2 距离 × 4 K，不加新候选）**:
- K ∈ {64, 128, 256, 512}
- 距离: standardized Euclidean（z 用 inner-train 月 StandardScaler）、cosine（同一标准化 z 上的余弦距离 1−cos）
- 权重: 高斯核 w_i = exp(−0.5·(d_i/d_K)²)，d_K = 查询点到第 K 近邻居的距离（确定性、良态）
- r_hat(q) = Σ_i w_i·r_i / Σ_i w_i
- 对每个候选: α 在 tune 月网格选择（最大化 cos(RMS(y0)+α·RMS(r_hat), y)），frozen 评估

**必须报告（每候选）**: delta cosine（主锚 + 辅锚）、corr(r_hat, y)、corr(r_hat, y−y0)、corr(r_hat, y0)、normalized MSE、4 outer、monthly delta、positive-month ratio、bootstrap CI（月级 5000 次, seed 2026）、neighbor distance 分布、邻居月份集中度（熵/最大单月占比）、β、选中的 α。

**P6R-00 继续门禁（→ P6R-01）**（全部满足）:
1. pooled corr(r_hat, r) > 0 且月级 bootstrap 95% CI 下限 > 0（至少 1 个 K ≥ 128 的候选）
2. 该候选的 tune 选中 α ≥ 0.10（不是 α→0 的噪声修正）
3. PSEUDO frozen delta > 0（主锚）
4. ≥ 3/4 outer 折 delta > 0 且无 catastrophic fold（worst > −0.0005）
5. K 鲁棒性: PSEUDO delta 在 ≥3/4 个 K 上同号
任一门禁失败 → 不进入 P6R-01；先做 failure analysis（state representation / retrieval metric / residual target / leakage）。

**附加诊断（不参与门禁）**: β=1 残差变体全量重算；V2 封顶银行重算；邻居距离与 delta 的关系（远邻居桶 vs 近邻居桶的月度 delta）。

## 4. P6R-01 — Local Varying-Coefficient Ridge（条件执行）

**算法（每查询 q, 月份 m）**:
- 银行 = V1 扩展（month < m）
- 特征 X: 40 个预注册核心特征（§6），scaler 只从 inner-train 月拟合
- 拟合加权 Ridge: β(z_q) = argmin Σ_i w_i(z_q)·(r_i − X_iβ)² + λ||β||²，w = P6R-00 高斯核
- r_hat_q = X_q·β(z_q)；y_hat = RMS(y0) + α·RMS(r_hat)
- **Global Ridge 对照**: 同一特征、同一银行全部样本等权单次拟合（lambda 同网格）——Local > Global 是 Conditional Alpha 的判定核心

**候选（预注册）**: K ∈ {128, 256} × λ ∈ {0.01, 0.1} × {Local, Global} = 8 个；α 网格同 §2（tune 选择）

**必须报告**: 每 outer: Local vs Global 的 tune/frozen delta；每核心特征: beta 全局 mean/std、beta by state（z 四分位桶）、sign stability（跨月同号率）、sign reversal 频率、beta vs 各 context 维度的相关；跨 fold 稳定性表。

**P6R-01 判定规则**:
- Local mean frozen delta > Global mean frozen delta（≥3/4 折同向）且 beta(state) 跨 fold 稳定 → **Conditional Alpha supported**
- 否则 → **Conditional Alpha evidence weak**，如实记录，不强行解释；按决策树降级/停止

## 5. 后续阶段（仅在前置正结果后逐级注册）

- P6R-02 anchors（n_anchor {16,32,64} 距离门控专家）← P6R-01 正
- P6R-03 OOD support gate（NN 距离/局部密度/局部残差方差/局部 beta 不稳定 → g(z)∈[0,1]）← P6R-02 后
- P6R-04 market latent（32d，需重训 P5-01 encoder）← P6R-03 gate 有效；必须 A/B/C 三臂
- P6R-05 surprise（5–60s vs 600s，05A market-only / 05B event-relative）← P6R-04 后
- P6R-06 FiLM / P6R-07 Soft MoE ← 逐级 gate；P6R-08 禁止

## 6. P6R-01 预注册核心特征（40 个，来自 f0726 schema 实际列名）

**OFI 族**: m_ofi_sum, m_ofi_sum_60, m_ofi_sum_180, m_ofi_weighted_300, m_ofi_ewm_120, x_m_ofi_long_short_diff
**trade imbalance / buy-sell pressure**: t_buy_sell_vol_ratio, t_avg_signed_vol, t_avg_signed_vol_30, t_large_buy_90, t_large_sell_95, x_large_trade_imbalance
**cancel pressure**: o_sec_cancel_new_ratio, o_sec_cancel_new_ratio_30, o_sec_cancel_volume
**aggressive flow**: o_market_ratio, o_market_ratio_30, o_sv_30, t_sv_weighted_15
**order/trade intensity**: o_vol_sum, o_sec_new_count, t_lv_mean, t_lv_mean_30, t_sec_vol_mean
**spread / depth / imbalance**: m_sp_mean_60, x_sec_cancel_spread, o_bid_depth, o_ask_depth, m_vol_weighted_60, m_imb_last, m_imb_mean_60
**microprice / price pressure**: x_vwap_mid_ratio, m_mid_last, m_mid_std, m_rv, m_rv_60
**execution / conversion**: x_trans_order_vol_ratio, x_tx_order_rate_ratio, x_trans_order_buy_diff
**pressure/depth + event velocity**: x_t_vol_weight_ratio_30, x_o_vol_weight_ratio_15, t_avg_time_gap, t_time_gap_std, x_t_gap_stability

（合计 6+6+3+4+5+7+5+3+4+3 = 46 — 从末尾族删 x_t_gap_stability、x_o_vol_weight_ratio_15、t_time_gap_std、o_sv_30、o_market_ratio_30、x_trans_order_buy_diff → **40 个**，最终名单以 §7 固定。）

## 7. P6R-01 最终 40 特征固定名单（执行前冻结）

OFI: m_ofi_sum, m_ofi_sum_60, m_ofi_sum_180, m_ofi_weighted_300, m_ofi_ewm_120, x_m_ofi_long_short_diff (6)
Trade: t_buy_sell_vol_ratio, t_avg_signed_vol, t_avg_signed_vol_30, t_large_buy_90, t_large_sell_95, x_large_trade_imbalance (6)
Cancel: o_sec_cancel_new_ratio, o_sec_cancel_new_ratio_30, o_sec_cancel_volume (3)
Aggressive: o_market_ratio, o_sv_30, t_sv_weighted_15 (3)
Intensity: o_vol_sum, o_sec_new_count, t_lv_mean, t_lv_mean_30, t_sec_vol_mean (5)
Spread/Depth/Imb: m_sp_mean_60, x_sec_cancel_spread, o_bid_depth, o_ask_depth, m_vol_weighted_60, m_imb_last, m_imb_mean_60 (7)
Micro/Price: x_vwap_mid_ratio, m_mid_last, m_mid_std, m_rv, m_rv_60 (5)
Exec/Conv: x_trans_order_vol_ratio, x_tx_order_rate_ratio (2)
Veloc/Pressure: t_avg_time_gap, x_t_vol_weight_ratio_30, x_o_vol_weight_ratio_15 (3)
**总计 40** ✓（全部为 f0726 schema 实际列名，代码内按名索引，找不到即报错）

## 8. 产物规范（每实验）

```
output/p6r_00/  (及 p6r_01...)
  predictions.parquet   # sample_id, month, y0, y, r, r_hat, y_hat, alpha, K, metric
  metrics.json          # 全部门禁指标
  monthly_metrics.csv
  fold_metrics.csv
  diagnostics.json
  config.json
  neighbor_diagnostics.parquet  # 每查询: K, mean_dist, max_dist, neighbor_month_entropy, bank_size
```

## 9. 泄漏防护清单（tests/ 强制）

1. query month m 不可能 retrieve month ≥ m（assert + 测试）
2. sample_id 唯一且正确对齐（canonical↔features↔baseline）
3. baseline prediction 是 OOF（source_train_end < month）
4. residual 与 label 对齐（同数组校验）
5. scaler 未 fit validation 月（只 fit inner-train 月）
6. anchor/邻居不使用 future 数据
7. α 不使用 frozen outer 选择（tune 月专用）
8. 输出预测数与 validation 行数一致
9. NaN/inf 不进入模型（全链路 assert）
10. 特征顺序固定（按名索引）
11. deterministic seed（2026）

## 10. 禁止清单（本轮）

Transformer/Mamba/大 RealMLP/大规模 HP 搜索/新 TCN/152 扩展/full multimodal concat/Soft MoE/Full World Model/M→O→T predictive coding/Kaggle submission/test target 推断/LB probing。P6R-08 不实施。
