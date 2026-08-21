# Codex Research Brief — MSCapital「交易行为隐状态建模 (BLSM)」路线执行任务

- 对象: Codex（独立研究执行）
- 日期: 2026-08-21
- 上游: AFAC2026「市场参与者交易行为识别」思路迁移评审 → 已通过 Hermes 查重 → 落账 `docs/blsm-proposal.md` + `docs/method-map.md`（BLSM 🧪 未测）
- 本 brief 自包含：不需要访问 Hermes 对话历史。所有需要的信息、纪律、验收标准都在下面。

---

## 0. 任务摘要（先读这段）

在已知的 Kaggle 比赛 **MSCapital — Real Financial Market Forecasting** 上验证一条新研究路线：

> **BLSM：不要把订单流压成单一不平衡值，而是从 Order + Transaction 序列学习「市场参与者正在执行什么行为」的连续隐状态 z_B，再结合 Market 600s 状态 z_M 判断该行为在当前环境下的含义，最终预测未来 10 分钟收益。**

**不要大规模训练。** 按严格 gate 顺序执行，目标是在每个 gate 用最便宜的手段回答"这条路值不值得继续"，负结果同样是正式结论。

---

## 1. 比赛与数据（Codex 需要知道的硬事实）

- **任务**: 预测未来 10 分钟收益方向。指标 **uncentered cosine，越大越好**。
- **数据**（本机 `<local-path>`）:
  - 原始三表: `raw/train/market.feather`（L1/L2 订单簿，600s）、`order.feather`（订单事件 60s）、`transaction.feather`（成交事件 60s）、`label.feather`（month 0-70 + target）
  - 处理特征: `processed/f0726_train_f32.parquet`（**152 基线特征**，1,257,637 行）、`f0726_test_f32.parquet`（647,896 行）
  - 项目自研 Z 特征: `f0726_train_z_f32.parquet` / `f0726_test_z_f32.parquet`（152+73Z）
- **验证协议（必须遵守，禁止改动）**:
  - PSEUDO: m0-32 训练 / m33-70 eval
  - **frozen 51-70 为审判窗**（结论只看这里）
  - 指标 uncentered cosine（大=好）
  - 单组件噪声线 ~±0.0005
- **运行环境**: 仓库 `<local-path>`（git）；venv `.venv/Scripts/python.exe`（polars≥1.43 / torch≥2.6）；**GPU 8GB 严格串行**（只跑一个 torch 任务）。

## 2. 项目现状（为什么 BLSM 时机合适）

已完成: 152 baseline / RealMLP / Z(SCFI +0.0075) / P9 归因闭环（撤单 ⊂ Z）/ 大量路线证伪。

**已证伪，不得重复**:
- 简单 OFI / 深浅 imbalance（与既有特征重合）
- O→T 点级 lag response（P8-01A: lag ±1/3/5/10s ≈ 噪声，0/71 月）
- 残差建模五连杀（residual target / retrieval / reconditioning）
- 幅度门控 / volatility confidence（P5-03 / P7-01 双重）
- Hard regime / categorical MoE（低优先，幅度路线已死）
- 原始事件时距统计 iat/burst（**P9-B Event-Time RED −0.0039**——152 基线已含事件节奏）
- 序列模型 test 分布外退化（TCN v6 灾难 0.082）
- 特征相对化 / 时间衰减加权 / 月度漂移预测

**已确认有效**: Z(SCFI) 条件创新（O−E[O|M]，+0.0070~+0.0087）、RealMLP、R2 归一化、152 基线。生产资产 = **152+73Z**。

## 3. BLSM 核心结构

```
Market State × Behavioral State → Price Impact → Future Return
z_M = MarketEncoder(M_600s)      # 市场环境: volatility/spread/depth/liquidity/trend/activity
z_B = BehaviorEncoder(O_60s, T_60s)  # 行为: aggressiveness/persistence/splitting/provision/withdrawal/absorption/resiliency
最终 X = [ X_baseline, z_M, z_B, z_M ⊙ z_B ] → ŷ = RealMLP(X)
```

**关键假设**: 行为本身未必有稳定 alpha，但 **Behavior | MarketState 可能有**。`z_M ⊙ z_B` 表达"同样的持续买入在高/低流动性下价格影响不同"。

**与既往路线的机制差异**（用于自证不是换名重跑）:
- vs OFI: OFI 是 X→scalar 单时不衡；BLSM 是 Sequence→State 过程描述
- vs O→T lag: 不假设 pointwise lag，sequence-level state 允许固定 lag 不存在
- vs categorical MoE: 连续 latent / soft membership，不做离散聚类
- vs residual: 不改 target，改 representation

## 4. 6 类行为 latent + **查重红线**（重要：防止重跑）

| 类 | 内容 | 状态 | 执行指示 |
|---|---|---|---|
| **B5 Absorption** | \|SignedFlow\|/\|PriceResponse\| 及其 buy/sell/bid/ask/trade/queue 变体 | 🆕 **零先例** | ✅ **最优先，G0 主对象** |
| **B6 Resiliency** | 冲击后 1s/3s/5s/10s recovery（depth/spread/mid/imb）| 🆕 零先例 | ✅ 与 B5 同列 G0 |
| **B3 Liquidity Prov/Withdrawal** | depth recovery / replenish / 单侧抽离 | ⚠️ 部分重叠 | 撤单侧已被 P9-A 闭合；**replenishment/补充维度** 0 条，可做 |
| **B4 Cancel Dynamics** | Cancel→Repost/PriceResponse 条件化（非单纯 CancelRatio）| ⚠️ 高覆盖风险 | 因 cancel ⊂ Z 已闭合，**除非条件化形式否则跳过** |
| **B1 Aggressiveness** | signed trade imb / active ratio / run length / burst | 🔴 与 P9-B RED 重叠 | **原始统计不重跑**；仅允许条件化变体 |
| **B2 Execution Persistence** | same-side run / size autocorr / interarrival | 🔴 同上 | **原始统计不重跑**；仅允许条件化变体 |

> Codex 必须先读 `docs/method-map.md`（BLSM 条目）+ `docs/failed-experiments.md` 再动手；任何想在 B1/B2 上"直接试"的冲动都先自查。

## 5. 强制执行纪律

1. **不重复已证伪方法**（查 method-map / failed-experiments / registry 三件套）
2. **不因 AFAC 教程声称"有效"就重跑**——迁移的只有"行为是隐藏状态"这一思想
3. 新方法先证明存在**增量信息**（G1 是裁决器，不是补充）
4. **必须用既有 frozen/temporal protocol**（frozen 51-70, uncentered cosine, 同 split 同 seed 同 loss）
5. **禁止在 validation 上反复调权后把结果当发现**（嵌套纪律）
6. 每个实验进 `experiments/registry.csv` + `docs/method-map.md`；负结果同规格保留
7. **大操作等授权**: 训练 >1h / 提交 / 动核心代码 → 停下汇报，等用户"开始"（本任务目标是 G0，不应触发 1h+ 训练）

## 6. Gate 序列（本任务的执行范围）

### ▶ 本 brief 的目标 = 完整跑完 G0（G1+ 只写方案，不开跑）

**G0 — Behavior Existence Gate（最便宜，无训练）**
- 输入: 从 O/T 提取 **30-50 个行为型统计量**（优先 B5 Absorption + B6 Resiliency 全族；可补 B3 补充维度）
  - experimental、persistence、burstiness、order splitting、absorption、replenishment、impact efficiency、resiliency、cancel dynamics
  - **注意**: 与 P9-B 的原始 iat/burst 必须区分——BLSM 统计的量是"对价格/深度/Spread 的响应关系"，不是单纯事件时距
- 方法（三选一 / 全做更佳）: G0-A PCA / G0-B GMM / G0-C 8-16 dim shallow autoencoder
- 必须回答 6 问:
  1. latent 是否稳定（跨月可复现）?
  2. 是否被 month 完全主导?
  3. 是否只是 activity proxy?
  4. 是否只是 volatility proxy?
  5. 是否只编码 order_count?
  6. 不同月份 latent 结构是否重复出现?
- **判 FAIL 条件**: embedding 与 order_count / transaction_count / m_mid_std / activity / month 高度等价（corr/AUC 说明冗余）→ 只是重新编码现有状态，记负结果终止。

**G0 验收产出**:
- 行为统计清单 + 构建脚本（入 `scripts/`）
- PCA/GMM/AE 稳定性与冗余诊断（数字）
- 结论: EXIST / FAIL（如实）
- registry 登记 BLSM-G0（completed 或 RED）

### ▶ G1 — Incremental Information Gate（只写方案+预注册门禁，不开跑）
```
A: Baseline152
B: Baseline152 + z_B
同 split/同 RealMLP/同 loss/同 seed/同 preprocessing, 唯一变量 = z_B
判定 (frozen): Δ<+0.0003 且无月度方向 → RED; +0.0003~0.0008 → YELLOW; >+0.0008 且多数月正 → 进 G2
```
报告里给出: 门禁写死、月度/activity/volatility/special-state bucket 报表设计。**禁止看到结果后改标准。**

### ▶ G2 — Interaction Gate（只写方案，不开跑）
```
A: base | B: +z_M | C: +z_B | D: +z_M+z_B | E: +z_M+z_B+z_M⊙z_B
E>D 且稳定 → 支持 Behavior×Market 交互假设（本路线最重要的新证据）
```

### ▶ G3 — Impact-State Gate（方案占位）
`z_I = ImpactState(Behavior→ObservedImpact)`，入选 `[z_M, z_B, z_I] → Y`。G2 成功后才考虑。

### ▶ SSL / 深层序列（明确禁止本轮动手）
自监督 encoder（masked event / next-event / contrastive）与 Transformer 序列模型：**仅当 G0-G2 全部证明方向后**再议。不用"规则→伪标签→XGBoost"（那是复制规则）。

## 7. 交付物

1. `docs/blsm-g0-report.md`: G0 完整结果（特征清单、诊断数字、EXIST/FAIL 结论）
2. `scripts/blsm_g0_*.py`: 特征构建 + 诊断脚本
3. registry 追加 `BLSM-G0` 行（28 列 CSV，格式参考现有行，编码 utf-8 无 BOM 追加）
4. git 提交（仓库 `<local-path>`），commit message 写明结论

## 8. 交接自检（完成前过一遍）

- [ ] 没有碰 B1/B2 原始统计
- [ ] G0 用 frozen/temporal protocol 而非自造切分
- [ ] 没有在 validation 上反复调权
- [ ] 没有开启 1h+ 训练（G0 无训练）
- [ ] FAIL 或 EXIST 都如实登记
- [ ] 全部产物已 git 提交
