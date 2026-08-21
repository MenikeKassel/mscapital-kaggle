# BLSM 路线提案 — Behavioral Latent State Modeling（交易行为隐状态建模）

> 来源: AFAC2026「市场参与者交易行为识别」思路对 MSCapital 的迁移评审（2026-08-21）
> 状态: **已加入 method-map（🧪 未测）**。执行前须逐关过 gate，禁止直接大规模训练。
> 设定: 机制假设 → 廉价 existence gate → incremental info gate → interaction gate → 才决定上深层序列。

---

## 1. 核心思想（与既有路线的边界）

- **不照搬 AFAC 的 OFI / XGBoost / LSTM / 硬规则伪标签**。
- 迁移的是「**交易行为作为隐藏状态**」的建模思想：
  ```
  Market State × Behavioral State → Price Impact → Future Return
  ```
- 学习连续行为隐状态 `z_B ∈ R^d`（主动执行强度/订单拆分/流动性提供/抽离/吸收/冲击效率/恢复速度/方向持续性/burstiness/补单），而非人类标签（游资/机构/量化）。

### 与既往路线的差异（防"换名字重跑"）
| 已有路线 | BLSM 的不同 |
|---|---|
| OFI (X→scalar) | BLSM 是 Sequence(O,T)→BehaviorState，描述一段过程而非单时不衡 |
| O→T lag (已证伪) | 不假设 pointwise lag，而是 sequence-level state（固定 lag 可不存在）|
| Hard regime / categorical MoE (已否决) | 不聚类成离散类别，用连续 latent / soft membership |
| Residual modeling (已证伪) | 不改 target，改 representation（X→z_B→Y）|

## 2. 数据分工（重定义）

- **Market 600s** → Market Context `z_M`（volatility/spread regime/depth/liquidity/trend/activity）：作为行为解释的上下文，不是再多几个 mean/std
- **Order 60s** → Intent（挂/撤/流动性提供/抽离）
- **Transaction 60s** → Realized Execution（方向/强度/集中度/持续性）
- **Label** → 学 (M_state, B_state) → Y，而非先入为主判定 bull/bear

## 3. 6 类行为 latent（现有查重结论见 method-map）

| 类 | 内容 | 三件套状态 |
|---|---|---|
| **B5 Absorption** | `\|SignedFlow\|/\|PriceResponse\|` 及其买/卖/bid/ask/trade/queue 变体 | ✅ **零先例，最优先** |
| **B6 Resiliency** | 冲击后 1s/3s/5s/10s recovery（depth/spread/mid/imb）| ✅ **零先例** |
| **B3 Liquidity Provision/Withdrawal** | depth recovery / replenish / 单侧抽离 | 🧪 补充维度未测；撤单侧 P9-A 已闭合 |
| **B4 Cancel Dynamics** | Cancel→Repost/PriceResponse 条件化（非单纯 CancelRatio）| ⚠️ 高覆盖风险（撤单 ⊂ Z）|
| **B1 Aggressiveness** | signed trade imb / active ratio / run length / burst | 🔴 与 P9-B Event-Time RED 重叠，原始统计不重跑 |
| **B2 Execution Persistence** | same-side run / size autocorr / interarrival | 🔴 同上，重叠 |

## 4. Gate 序列（强制顺序，禁止跳跃）

- **G0 existence gate**（最便宜）: 30-50 个行为统计 → PCA/GMM/8-16dim AE → latent 是否稳定/是否被 month/activity/volatility/order_count 主导。若 embedding ≈ 现成状态 → **FAIL**。
- **G1 incremental info gate（核心）**: Baseline152 vs Baseline152+z_B，同 split/同 RealMLP/同 loss/同 seed。判 frozen Δ：<+0.0003 且无月度方向 → RED；+0.0003~0.0008 → YELLOW；>+0.0008 且多数月正 → 进位。**禁止事后改标准**。
- **G2 interaction gate（核心）**: A=base / B=+z_M / C=+z_B / D=+z_M+z_B / E=+z_M+z_B+z_M⊙z_B。E>D 且稳 → 支持"Behavior Effect 取决于 MarketState"。
- **G3 impact-state gate**（G2 成功后）: 显式建模 Behavior→ObservedImpact（trade→mid/depth response, shock→recovery, flow→spread response）→ z_I → [z_M,z_B,z_I]→Y。
- **SSL 自监督 encoder 与深层 Transformer**: 仅在 G0-G2 证明方向后考虑；不用 AFAC 式"规则→伪标签→XGBoost"。

## 5. 执行纪律（与项目纪律一致）

1. 不重复已证伪方法（B1/B2 原始统计与 P9-B 重叠，须条件化或跳过）
2. 不因 AFAC 教程声称"有效"就重跑
3. 新方法先证明存在增量信息（G1 是裁决）
4. 必须用既有 frozen / temporal protocol（frozen 51-70, uncentered cosine）
5. 禁止在 validation 反复调权后当发现
6. 每个实验进 registry + method-map；负结果同样正式保留

## 6. 复现入口

- 提案: 本文件（docs/blsm-proposal.md）
- 查重: docs/method-map.md（BLSM 条目）+ docs/failed-experiments.md + experiments/registry.csv
- 目标产物: G0 后决定是否立项；若立项，特征构建入 scripts/，实验入 P 系列 registry
