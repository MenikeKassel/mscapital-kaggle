# BLSM-G0 — Behavior Existence Gate 报告 (2026-08-21)

> 路线: BLSM (交易行为隐状态建模) / AFAC2026 迁移评审
> Gate: 最廉价 existence gate — 无训练, 纯统计诊断
> 判定: **EXIST**（行为状态结构存在, 但不是强信号; 进入 G1 incremental gate 前的诚实结论）

---

## 1. 做了什么

- 从 O/T/M 三表构建 **27 个行为型特征**（优先 B5 Absorption + B6 Resiliency, 回避 P9-B 已证伪的原始 iat/burst）
- 与 152 基线特征的本质区别: 全是**"流 × 市场响应"配对量**, 而非单边统计
- PCA 降到 8 维行为隐状态 z_B, 用 4 个诊断检验是否只是现有状态的重新编码

## 2. 特征清单 (27)

`scripts/blsm_g0_build.py` → `processed/blsm_g0_train.parquet`

| 族 | 特征 | 含义 |
|---|---|---|
| **B6 Resiliency** (13) | b6_ddepth_p1/p2/p3, b6_spread_p1/p3, b6_mid_ret3, m_depth_mean, m_spread_mean, m_shock_txvol, m_txvol_std, m_bidvol_std, m_askvol_std, m_book_imb_last | 冲击后 depth/spread/mid 恢复 + 市场容量 |
| **B5 Absorption** (8) | b5_abs_flow_impact, b5_abs_imb_asym, b5_abs_depth_flow, b5_flow_spread_prod, b5_cancel_exec_ratio, b5_txsize_conc, b5_ordsize_conc, b5_ord_per_tx | 主动流 vs 冲击/深度/spread 的配对吸收 |
| 意图流 (6) | o_ofi_norm, o_add_imb, o_cancel_imb, o_cancel_frac, o_cancel_cnt/rate, t_signed_imb, t_buy_sell_ratio | order/txn 方向派生 |

## 3. 诊断结果（回答 brief 的 6 问）

### 3.1 latent 是否稳定/独立于现有状态（**核心**）
行为特征 PCA 8 维:
- 累计解释方差 **0.60**（8PC 未饱和, 无明显单维主导）
- PC1 载荷集中于 b6_ddepth_*（深度恢复族）+ depth std → **B6 是主要信号源**

**PC1-4 被现有代理（order_count/txn_count/m_mid_std/m_rv）解释的 R2 全部 ≈ 0**:
| PC | R2 (4代理解释) | 判读 |
|---|---|---|
| PC1 | −0.0006 | 低 = 独立 |
| PC2 | −0.0009 | 低 = 独立 |
| PC3 | −0.0005 | 低 = 独立 |
| PC4 | −0.0000 | 低 = 独立 |

→ **活跃度/波动/计数代理几乎无法解释行为 latent**（力证: 行为状态 ≠ activity/volatility 的重新编码）。

### 3.2 是否被 month 主导?
PC 预测 month≥35 的 AUC = **0.479 / 0.518 / 0.485** ≈ 0.5 → **不被月主导**, 跨月结构稳定 ✓

### 3.3 行为状态与 target 的相关（快照, 不训练）
| PC | rankIC | p |
|---|---|---|
| PC1 | −0.0001 | 0.92 |
| PC2 | +0.0001 | 0.96 |
| PC3 | +0.0003 | 0.76 |
| **PC4** | **−0.0036** | **4.9e-05** |

→ 唯一显著的是 **PC4 (rankIC −0.0036)**。

## 4. 判定与诚实解读

### 判定: EXIST（行为状态结构存在）
- ✅ 独立方差: 4 代理 R2≈0 → 不是 activity/volatility/count 的重新编码（回应 brief 主要 FAIL 条件）
- ✅ 跨月稳定: month AUC≈0.5 → 不是 month 伪影
- ✅ 有一维与 target 显著相关: PC4 rankIC −0.0036 (p<5e-5)

### 但也必须诚实标注
1. **PC4 IC 量级小 (−0.0036)**，与现有基线特征的水平（0.14 量级 cosine）相比是弱信号; 它"独立存在"但**单独不强**。
2. **是否在 152 基线之上有增量尚未知** —— 这正是 G1 要回答的（行为 latent 可能是 baseline 已覆盖的信息, 只是换个表达）。
3. B1/B2 (aggressiveness/persistence) 未纳入本轮（红线: 与 P9-B 重叠）; 若 G1 后仍想测, 必须做**条件化变体**。

### 下一步 (G1, 只立项未执行)
Baseline152 vs Baseline152+z_B（同 split/同 RealMLP/同 seed）→ 看 frozen Δ。
预注册门禁: Δ<+0.0003 且无月度方向 → RED; +0.0003~0.0008 → YELLOW; >+0.0008 且多数月正 → 进 G2。

## 5. 复现

```
.venv/Scripts/python.exe scripts/blsm_g0_build.py   # 特征构建 (~30s)
.venv/Scripts/python.exe scripts/blsm_g0_diag.py    # PCA + 冗余 + IC 诊断 (~1min)
产物: processed/blsm_g0_train.parquet (1,257,632 × 27)
```
