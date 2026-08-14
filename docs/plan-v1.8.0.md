# MSCapital Plan v1.8.0 — GPT1 四论 + GPT2 一论吸收 (条件化/创新信息收敛)

> 日期: 2026-08-14 | 版本: v1.8.0 (上一版 v1.7.0)
> 来源: 用户转述 GPT1 四论 (State-Conditioned Residual MoE) + GPT2 一论 (Conditional Innovation) + 本地 P5-02M 幅度调制提案
> 状态: **讨论定稿阶段, 未获执行许可** — 等用户拍板

## 版本历史
- v1.8.0 (2026-08-14): 吸收 GPT1 四论 + GPT2 一论 — 三方案独立收敛到同一核心 ("条件化/创新信息"); 8 篇论文全部核验 ✅; 关键统一认识 (Multi-scale Surprise = Conditional Innovation 的线性简化); 队列合并为 P5-02M → B-lite v2 → P6-04 渐进; P6-01 降级 (与 P5-03 同义, probe 预判死)。
- v1.7.0 (2026-08-14): 吸收 GPT 三论 — 六信息层框架; 修正 4 处; P5-02B-lite 审计前置。
- v1.6.0 (2026-08-14): P5-02I 执行完毕 — 信息结构五条实锤; GPT1 幅度猜想反转; P5-03 降级; backbone 建议。
- v1.5.0 (2026-08-14): GPT 二轮 — P5-02I 前置; 2 处修正; 论文核验 2 篇。
- v1.4.0 (2026-08-14): GPT 首轮 — Alpha Factory; 3 处修正。

## 0. 两份新方案 + 本地 P5-02M 的收敛分析 (本版核心)

### 三方案独立收敛到同一思想

| 方案 | 核心 | 实现层次 | 成本 |
|---|---|---|---|
| GPT1 四论 | State-Conditioned Residual MoE: ŷ = f_152(Z) + Σ_k Gate_k(M)·Expert_k(O,T,M); 双残差化 (target + feature); Impact = Pressure×Conversion/Capacity; Multi-scale Surprise | 特征层 + 结构层 (MoE gate) | 中 |
| GPT2 一论 | Conditional Innovation: U_M = M_recent − E[M_recent\|M_bg], U_O = O − E[O\|M], U_T = T − E[T\|M,O]; H = [Z_152, Z_M, U_M, U_O, U_T] → RealMLP | 表示层 (条件模型) | 高 |
| **本地 P5-02M** | 幅度调制: pred' = v7 × (m_i/median)^γ, m_i = market 幅度预测 (已有模型) | **融合层 (零新训练)** | **极低** |

**共同核心**: "实际 − 给定状态/上下文的期望" (surprise/innovation/条件化)。市场决定状态/期望, order/tx 的价值在"相对状态的异常", 不在绝对值。三份独立方案 (两个外部 AI + 本地证据驱动) 收敛到同一方向 = 该方向值得认真对待。

### 关键统一认识 (实测推导)
**GPT1 的 Multi-scale Surprise (z-score 形式) = GPT2 Conditional Innovation 的线性简化版**:
```
U_X ≈ z = (x_recent − μ_600) / σ_600   (5/10/20/30/60s vs 600s)
```
z-score 版零条件模型 (半天审计), 条件模型版 (E[O|M] 神经网络) 工程量大且订单流条件预测本身噪声大 → **先审计 z-score 版, 有效再上条件模型**。GPT2 的 U_O 简化 = surprise 家族的 order 版, U_T 简化 = transaction 版 — 已被 Surprise 家族覆盖。

### P5-02M 与两份方案正交
- P5-02M = **融合层条件化**: 用 market 幅度条件化 v7 的逐样本权重 (D1-D3 诊断已支持: v7 方向正确率随 |y| 单调 0.37→0.59, 内积 74% 集中 top20%)
- Surprise/MoE = **特征/表示层条件化**: 用 market 期望条件化特征本身
- 两者不冲突, 可并行; P5-02M 30 分钟先跑 (零成本验证"条件化信息在 cos 指标上能变现"), 其结果是 B-lite/MoE 投入决策的前置证据

## 1. 论文核验 (8 篇全部通过, export API 实测)

1011.6402 ✅ Price Impact of OBE (Cont 2010) | 2502.15757 ✅ TLOB | 2505.02139 ✅ LOB Repr | 2511.12563 ✅ LOBERT | 2607.09230 ✅ When Does Order Flow Matter (2026-07) | 2604.23961 ✅ State-dependent Hawkes (2026-04) | 2602.23784 ✅ TradeFM (2026-02) | 2502.17417 ✅ Event-based LOB Sim (2025-02)

## 2. 与本地证据核对 (支持面 vs 风险面)

### ✅ 支持面 (条件化思想有实锤)
- R2 归一化 (绝对→相对) LB 0.122→0.123, 远期 fold +0.0023 — "相对状态"方向已验证
- P5-02I: low activity 增益 lo +0.0021 vs hi +0.0006 — **regime 条件化增益的直接证据**
- P5-02I: market = 状态/幅度信息源 (extreme AUC 0.78, |y| corr 0.43) — market 适合做 gate/条件 的前提成立
- D1-D3 诊断 (本版新增): v7 方向正确率按 |y| 分层 0.37→0.585 单调, 内积 top20% 占 74% — 幅度调制的机制前提成立
- GPT1 引 2607.09230: "state first, flow additive and conditional" — 与本赛低活动条件化增益同构

### ⚠️ 风险面 (实现形态决定成败, 必须审计先行)
- **P4-01a: 演化 ratio 类特征 (vol/spread/depth_recent_early, mid_trend) 区分力 ≈ 0** — surprise 的粗版本 (recent vs early 比值) 无信号; 但那是 LB142 分歧画像 (解释模型行为), 不是 target 残差, 且 ratio 形式 ≠ z-score 形式 — **需实测区分**
- P4-06A: 聚合 600s 状态条件化 → y−v7 残差 = +0.0000 (链条断裂)
- P3-02: latent 拼接 +0.00095 gate F (拼接不叠加)
- P5-02I resid probe: market 序列 → y−β·v7 方向 AUC 0.51 ≈ 0.5

## 3. 队列 v1.8.0 (三方案合并, 渐进式)

| 优先级 | 实验 | 来源 | 内容 | 门禁 |
|---|---|---|---|---|
| **S** | **P5-02M 幅度调制** | 本地 (D1-D3 驱动) | 重训 P_mag 保存 m_i (5min) → pred' = v7×(m_i/median)^γ, γ 网格, 41-50 选, 51-70 冻结; 对照 f 单调族 (幂/分位门控/线性) | 51-70 Δ > 0 且月度 > 13/20 才升级双头模型 | 零新架构; 也是 gated alpha (P5-05) 预演 |
| **S** | **P5-02B-lite v2 跨源审计** | GPT1 四论 + GPT2 一论交集 | Surprise 家族 (5/10/20/30/60s z-score vs 600s, market+order+tx 各维度) + Structural 家族 (Pressure/Conversion/Capacity/Impact= P×C/Q) + O×T 转化率; 相对形式; **双 target: y 残差 + \|y\|**; 4.7 双轴 (corr vs 152 + Δcos + ΔAUC) | 双轴: corr<0.70 且 ΔPSEUDO≥+0.0015 (对 y 或对 \|y\| 任一) | 半天; z-score 版 = innovation 线性简化; 通过才上条件模型/MoE |
| **S** | **P6-04 hard regime experts** | GPT1 四论 (P5-02I 分层证据) | 按 market 状态 (活动度/波动分层) 分 3-5 个 regime, 各自训练 residual learner | 证明 f_regime1 ≠ f_regime2 (同输入不同映射增益) | hard 先行, soft MoE 后置 |
| A | P6-05 soft MoE | GPT1 四论 | learned gate Σ p_k(M)·r_k | P6-04 通过后 | — |
| A | conditional innovation 条件模型 | GPT2 一论 | U_O = O − E[O\|M] 等 (神经网络条件模型) | B-lite v2 的 z-score 版通过后 | 工程量大; 订单流条件预测噪声大 |
| B | P6-01 residual target | GPT1 四论 | 冻结 v7 OOF, 全输入 → y − ŷ_152 | 同 P5-03 | **⚠️ 与 P5-03 同义: resid probe AUC 0.51 预判死, 降 B** |
| B | U_152 (152 对 market 条件化) | GPT2 一论 | Z_152 − E(Z_152\|M) | lite 通过后 | 工程量大 |
| B | SSL 预训练 (M→recent, M→O, M,O→T) | GPT1+GPT2 共同 | 掩码/预测自监督 → fine-tune | 第二阶段 | 两方案都同意放后 |

## 4. 停止清单 (继承 + 新增)

继承 v1.7.0 全部。**新增**: 条件模型/MoE/SSL 不经 B-lite v2 审计直接展开 (收敛思想必须先过 z-score 审计); P6-01 不经 resid probe 复查直接投入 (已预判死)。

## 5. 待用户拍板

1. 三方案收敛判断 + 合并顺序 (P5-02M 30min → B-lite v2 半天 → P6-04 hard experts) 接受否
2. P6-01 降级 B (与 P5-03 同义, probe 预判死)
3. **是否开始执行 P5-02M** (零成本, 30 分钟出冻结验证结果)


## 0. GPT1 三论框架摘要 (六信息层)

| 信息层 | 来源 | 含义 | 本地证据 |
|---|---|---|---|
| State | Market | 盘口结构/流动性/脆弱性 | ✅ P5-02I: market = 波动/幅度状态 (extreme AUC 0.78), 非方向 |
| Intent | Order | 参与者意图 | ✅ 152 特征主体 (人工提炼) |
| Realization | Transaction | 意图兑现 | ✅ 152 特征主体 |
| Expert | 152 | 已验证人工 alpha | ✅ LB 0.135 主力, 绝不废 |
| Surprise | 600s vs 60s | 事件 − 背景期望 | ⚠️ 部分验证 (R2 归一化 +0.0023), 细粒度差值未测 |
| Interaction | M×O×T | 状态下事件的真实意义 | ⚠️ 有负面先例, 需门禁 (见修正 2) |

GPT1 主张: 生产架构应联合建模 P(Y|M,O,T) = f(State, Intent, Realization, Surprise, Interaction), 三 encoder → C/I/F → 交互层 (C×I, C×F, I×F, C×I×F) → 拼 152 → 多头 (cosine + sign + mag + rank + extreme multitask)。

## 1. 逐条核实表 (对照本地台账 + P5-02I 新证据)

| GPT1 论断 | 本地证据 | 判定 |
|---|---|---|
| Market 是 Context 不是独立预测器 | P5-02I: market 信息 = 幅度/波动状态 (mag 0.43, extreme AUC 0.78, sign 仅 0.56) | ✅ 强一致 (P5-02I 刚实锤) |
| 152 保留为 expert branch, raw+expert 结合 | 152×RealMLP = LB 0.135 主力; Attribution 特征×架构组合 | ✅ 采纳 |
| 相对量/Surprise 归一化 | 4.6 R2: 绝对→相对 LB 0.122→0.123, 4/4 folds +0.0023 | ✅ 方向契合 (但 R2 是"绝对→相对", 不是"事件−历史期望"差值, 后者未测) |
| 时间对齐/统一时间轴 | M7 跨源错位 = 本队列已有审计项 | ✅ 采纳 (P5-02B 门禁) |
| label 多成分 → multitask | P5-02I probes: sign 弱 mag 强, 乘法双头 = 直接实现 | ✅ 方向一致, 但生产化要过纪律 (修正 4) |
| **分源研究是错的, 应直接联合** | P5-01/P5-02I 恰恰靠分源隔离才定位信息 | ⚠️ 修正 1 (阶段混淆) |
| **三阶交互 M×O×T 是最大未知** | P3-02 latent 拼接 +0.00095 gate F; P4-02 自动交互 127 个无增量; P4-06A 状态条件化 600s→残差 0 | ⚠️ 修正 2 (负面先例, 需门禁) |
| **600s context 分层 (hierarchical)** | P5-02I: market 序列内部信息 ≤30s (block10 崩); P4-06A: 聚合 600s 状态对残差无信息 | ⚠️ 修正 3 (长上下文权重下调) |
| **multitask loss 直接生产化** | 4.14: lambda_cos=0.01 名义混合实际 99% MSE; 两次 Public 0.135 | ⚠️ 修正 4 (受控消融先行) |

## 2. 四处修正 (GPT1 三论与本地证据冲突)

### ⚠️ 修正 1: "分源研究是错的" = 阶段混淆
研究方法必须继续分源隔离 (P5-01/P5-02I 刚证明: 只有隔离单源才能定位"信息在序列不在聚合、在短窗、在跨通道、在幅度")。GPT1 批评的合理对象是**生产架构** (不能只拼单源独立预测器), 不是**研究期分源审判**。两个阶段方法论不同, 不冲突。文档明确: 研究 = 分源隔离; 生产 = 联合建模。

### ⚠️ 修正 2: 三阶交互有本地负面先例, 必须预注册门禁
- P3-02: 状态 latent + 事件流 61 特征**拼接** PSEUDO +0.00095 (gate F, 拼接不叠加反而低于纯状态)
- P4-02: LB142 factors 127 个自动交互特征 = 与 M04 交互同测无增量 → "信息差基本排除"
- P4-06A: 28 个 4段×7 market-state 特征条件化残差 PSEUDO +0.0000 (链条断裂)
**结论**: "交互"不是构造了就一定有信息。GPT1 交互层只有在 **latent 层面让模型自己学** (而非手工交互特征) 时才有独立测试价值; 且必须先过 **M7 对齐 vs 错配门禁** (对齐 Δ 显著 > 错配 Δ 才证明跨源 synergy 真实存在)。手工交互特征家族 (O×T 转化率等) 降为轻量审计 (P5-02B-lite), 4.7 双轴筛选。

### ⚠️ 修正 3: 长上下文 (600s) 权重下调 — P5-02I 新证据
P5-02I 实锤: market 序列内部有效信息 ≤10 步 (30s), block50 与 block10 无差别; 且 P4-06A 显示聚合 600s 状态对 y−v7 残差无信息。GPT1 的 "long context → medium event → short trigger" 三层里:
- **600s context 层预期低**: 它只能做"状态条件化" (其信息上限已被 P4-06A 证伪过一次), 不是长程时序
- **真正增量预期在事件层交互** (60s order/tx × 30s market trigger)
- Surprise 特征 (事件−背景期望) 的价值 = 归一化防漂移 (R2 已验证方向) + 条件化, 不是新的长程信息

### ⚠️ 修正 4: multitask 生产化纪律 (4.14 教训直接适用)
- 生产 loss 权重审计: GPT1 的 λ1..λ5 多任务 loss 直接上生产 = 重演 lambda_cos=0.01 事故 (名义混合实际单任务)
- 纪律: 辅助任务先做**受控消融** (P4-08A 协议: 同输入/同架构/同切分, 一次一个任务, 4 outer), 验证增益再进生产
- 乘法双头 (P5-02I probe 直接驱动) 与 multitask 加权二选一对比, 不叠加
- 提交命题纪律不变: 验证配方 = 生产配方

## 3. 队列 v1.7.0 (GPT1 三论吸收后)

| 优先级 | 实验 | 内容 | 门禁 | 备注 |
|---|---|---|---|---|
| **S** | **P5-02B-lite 跨源交互审计** (新, GPT1 三论的最小可测形式) | 特征级: O×T 转化率 (order pressure − realized pressure, cancel-to-trade, aggression conversion) + M×O 冲击比 (OFI/depth, imbalance×inv-depth) + M×T 深度比 (vol/depth, signed flow×spread) + Surprise (60s 事件 − 600s 背景期望, 相对形式)。全部相对/归一化, 4.7 双轴筛选 (corr vs 152 + Δcos + ΔAUC) | 双轴: corr(新特征,152)<0.70 且 ΔPSEUDO≥+0.0015 才展开; 否则交互假说证伪, 不开大模型 | 快: 复用 f0726 + market 聚合, 半天内; 直接检验 GPT1 "差值信息" 假说 |
| **S** | **P5-02B-latent 三源联合** (GPT1 总架构, 审计通过后) | 三 encoder (market 短窗 channel-mixing ≤100步 / order / tx 秒级聚合, P1-2 基建复用) → C/I/F → 交互层 (C×I, C×F, I×F, C×I×F, latent 层面) → 拼 152 → cosine 头 | **M7 对齐 vs 错配先过** (对齐 Δ > 错配 Δ 显著); 再 corr(latent,152)<0.70 且 ΔPSEUDO≥+0.0015 | 工程量大; 等审计 + 用户拍板; 架构按 P5-02I 约束: 短窗/channel-mixing/无方向性 |
| **S** | **P5-02 序列 latent** (主线, backbone 已定) | 短窗 (60-100步) channel-mixing encoder → 32d latent → 拼 152 → RealMLP | corr(latent,152)<0.70 且 ΔPSEUDO≥+0.0015 | MLPLOB-lite 风格; 可先于 B-latent 跑 (单源, 便宜) |
| **S** | **P5-02 乘法双头** (P5-02I probe 驱动) | pred = sign_head × mag_head, cosine 目标, 与单头对照 | 同 P5-02 门禁 | 幅度富矿 0.43 被符号瓶颈 0.56 卡住; multitask 加权版降 B 级对比 |
| **S** | **P5-02B context×event** (v1.6.0 继承, 并入 B-lite/B-latent) | 600s context × 60s event | 见 B-lite/B-latent | 被 B 系列吸收, 不再单列 |
| B | multitask loss 受控消融 | cos+rank+sign+mag+extreme 各 λ 消融 | P4-08A 协议, 4 outer | 4.14 纪律; 与乘法双头对比不叠加 |
| B | P5-03 序列版残差 | (P5-02I resid probe AUC 0.51 预判死) | — | 保留仅作可选验证 |

## 4. 停止清单 (不变, 继承 v1.4.0)

TCN 大规模调参 / vanilla Transformer / 152 cosine 花式变体 / 单纯 RealMLP 参数搜索 / 换皮 / calibration / ensemble 套娃 / 34 聚合特征重做 / 月漂移后处理。**新增: 手工交互特征不经双轴审计直接展开 (修正 2); multitask 不经消融直接进生产 (修正 4)**。

## 5. 待用户拍板

1. 四处修正 (分源研究阶段论 / 交互门禁 / 长上下文降权 / multitask 纪律) 接受否
2. **P5-02B-lite 跨源交互审计是否先行** (半天成本, 直接检验 GPT1 "差值/交互信息" 假说; 通过才开 B-latent 大模型)
3. P5-02 单源两条 (latent / 乘法双头) 与 B-lite 的优先级: 建议顺序 = 乘法双头 (最便宜, probe 已实锤) → B-lite 审计 → latent / B-latent
4. 确认后是否开始执行


## 0. P5-02I 执行结果摘要 (2026-08-14, 全套结果见 docs/p5-02i-info-audit-report.md)

FROZEN 51-70, R = 相对 M0 损失:

| 臂 | corr(y) | R(corr) | frozen Δ | R(Δ) | 判读 |
|---|---:|---:|---:|---:|---|
| M0 raw | +0.0861 | 0 | +0.000926 | 0 | 基线 (P5-01 复现 ✓) |
| M1 shuffle | +0.0231 | 0.73 | +0.000155 | 0.83 | 时序必要 |
| M2 reverse | +0.0852 | 0.01 | +0.000929 | ~0 | **无时间箭头** |
| M3-5 block10/20/50 | ~0.026 | 0.69-0.70 | ~0.0002 | 0.69-0.81 | **信息尺度 ≤10 步 (30s), 无长程** |
| M6 desync | +0.0197 | 0.77 | +0.000057 | 0.94 | **跨通道同步是核心** |
| M7 phase | +0.0038 | 0.96 | −0.000256 | 1.28 | **相位/非线性形态** |

**五条实锤**: 时序必要 / 无时间箭头 / 短程 ≤10 步 / 跨通道同步核心 / 非线性相位形态。
**一句话**: alpha = 短时窗 (≤30s) 内跨通道同步的、时间反演对称的非线性事件形态。

**Probes**: sign AUC 0.56 (方向弱) | rank corr 0.089 | **|y| corr 0.43-0.47 (幅度巨大)** | extreme AUC 0.78 (大波动识别强) | **y−β·v7 方向 AUC 0.51 ≈ 0.5 (残差无信息)**。

**⚠️ GPT1 猜想反转**: "sign 大/magnitude 小" 实测相反 — 幅度是富矿, 方向是瓶颈。MSE 失败机制 = y=sign·|y| 且 E[sign|x]≈0 → MSE 最优解≈0; cosine 兑现弱方向。**乘法双头 pred = sign_head × mag_head** 有独立增益空间; market 幅度 = gating 信号。

## 1. GPT 二轮核心论点 (与本地证据核实)

GPT 二轮: "问题不再是 market 有没有信息, 而是哪一种信息在聚合过程中被毁掉" — 用信息论表述:

- 已证: `I(Y; X_m | Z) > 0` (P5-01: corr(y)=0.086, corr(v7)=0.49, frozen Δ=+0.0009)
- 已证: `I(Y; g(X_m) | Z) ≈ 0` (34 聚合特征 PSEUDO −0.0002; 4段×7 state 残差 long +0.0000; 演化 ratio 特征区分力≈0)
- 推论: **有用信息存在于 market 序列, 不在低维充分统计量中** → 与 v1.4.0 修正 1 完全同向 (GPT 已放弃手工 feature bank, 转向"信息定位"再定架构)

| GPT 论断 | 本地证据 | 判定 |
|---|---|---|
| 同分布不同顺序 (mean/std 不变, 市场意义不同) → temporal ordering 是第一嫌疑 | P5-01 序列有信号、聚合无信号, 机制上自洽 | ✅ 采纳为 M1/M2 动机 |
| transition / cross-channel sync / 跨源 synergy / event-time / higher-order 六类候选信息 | 与 4.15 的"信息在水平/密度/波动幅度"初步画像互补 (那是对 LB142 分歧的画像, 非 target 残差) | ✅ 采纳, 用 surrogate 实测 |
| surrogate data test (M0-M7) 优于继续堆模型 | 与用户"问题驱动、禁机制清单惯性试错"偏好一致 | ✅ 采纳为核心方法 |
| target 分解: I(X_m;sign(Y)) 大 / I(X_m;\|Y\|) 小 → 解释 MSE 失败 cosine 成功 | P5-01 MSE corr −0.0013 vs cosine +0.086, 方向自洽 | ✅ 采纳为诊断 probe |
| gated alpha: pred = v7 + α(x)·market (low activity 最强 → conditional) | P5-01 lo +0.0021 > hi +0.0006 实锤条件性 | ✅ 采纳为 P5-05 可选形态 (预注册) |

## 2. 验证记录 (本版实测)

1. **arXiv 论文核验** (export API id_list 单请求): 2502.15757 = TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction with LOB Data ✅; 2505.02139 = Representation Learning of Limit Order Book: A Comprehensive Study and Benchmarking ✅ (GPT 引用真实, 标题一致)。
2. **P5-01 数字复核**: 台账与脚本一致 — corr(y)=+0.086, corr(v7)=0.492, frozen 51-70 Δ=+0.0009 (17/20 月正), lo +0.0021 / hi +0.0006。GPT 引用无失真。
3. **P5-02I 成本实测评估** (p5_01_market_sequence.py): EPOCHS=15, BATCH=1024, train 21-40 ≈18 万样本, 小残差 Conv1D (64ch k=7) → **单臂训练分钟级, 8 臂合计 ~1-2h 可行**。M1-M4 (shuffle/reverse/block/desync) 可用**索引变换 on-the-fly** (零存储, 不物化数据); M5 (phase randomize) 需一次 FFT 预处理 (~分钟级)。
4. **输入结构**: 200 步 × 18 通道, 均匀 3s 网格 (0..597s, carry-forward 最近快照) — 见修正 A。

## 3. 两处修正 (GPT 二轮技术缺陷, 实测后纠错)

### ⚠️ 修正 A: M6 (uniform resample / event-time) 在 P5-01 协议上不可行
P5-01 序列**已经是均匀 3s 网格重采样** (0..597s, carry-forward) — 原始 event-time 结构 (非均匀快照秒数) 在构建时已被抹平, "uniform resample" 无差异可测。event-time 信息需要回到原始快照 seconds (0-600 非均匀) 重新构建 → **M6 标记 SKIP**; 若要做 = "非均匀 vs 均匀重采样" 对比, 属 P5-02B 范畴 (条件项)。

### ⚠️ 修正 B: M7 (market↔event 错位) 需要双源模型, 不是 market-only Conv1D
P5-01 模型**不看 order/transaction** (纯 market 序列), 错位无从构造。M7 必须在 context×event 模型 (P5-02B) 上做: 正确对齐 vs 样本错配 (market_i × event_j≠i) 对照。**M7 移到 P5-02B, 作为其 surrogate 审计门禁的一部分** (GPT 原设计"同一 Conv1D"仅对 M0-M5 成立)。

## 4. 执行队列 v1.5.0 (GPT 二轮吸收后)

| 优先级 | 实验 | 内容 | 门禁 | 备注 |
|---|---|---|---|---|
| **S** | **P5-02I Information Audit** (新, GPT 二轮) | M0 RAW 基线 + M1 全随机 shuffle + M2 time reverse + M3 block shuffle 10/20/50 + M4 channel desync + M5 phase randomize。同一 Conv1D/cosine/uncentered/同 seed/同切分 (21-40 → 41-50 选 α → 51-70 冻结)。每臂报: corr(y) / corr(v7) / frozen Δ / 月度正率 | 诊断实验, 无数字门禁; **结果决定 P5-02 架构** | M6 SKIP (修正 A); M7 → P5-02B (修正 B); M1-M4 索引变换零成本, M5 FFT 预处理; 复用 p5_01 脚本骨架, 一次跑完 |
| **S** | **P5-02 序列 latent** (主线, backbone 待拍板) | frozen market encoder → 32d latent → 拼 152 → RealMLP | corr(latent,152)<0.70 且 ΔPSEUDO≥+0.0015 | **backbone 按 P5-02I 定: 短窗 (60-100 步) + channel-mixing 优先 (MLPLOB-lite 风格 Conv1D 短核 + per-step channel MLP); 淘汰长注意力/长 TCN/方向性 RNN** |
| **S** | **P5-02 乘法双头** (新, P5-02I probe 驱动) | pred = sign_head × mag_head, cosine 目标 | 同 P5-02 门禁, 与单头对照 | 幅度富矿 (0.43) 被符号瓶颈 (0.56) 卡住, 双头是唯一同时利用两者的形态 |
| **S** | **P5-02B context×event** (含 M7) | 600s context × 60s event 相对/交互特征 → RealMLP/cosine; M7 错位对照 (对齐 vs 错配) | corr 双轴诊断先行; 展开后 ΔPSEUDO≥+0.0015 且 ≥3/4 折正; M7: 对齐 Δ 显著 > 错配 Δ | 全部相对形式 (防漂移); M7 结果 = 跨源 synergy 直接证据 |
| **S** | **P5-03 序列版残差** ⚠️ 降级 | P5-01 cosine 序列模型 → 目标 r = y − β·v7 | corr(res_pred,y)>0 且 corr(res_pred,v7)≈0 且 frozen Δ>0 | **P5-02I resid probe AUC 0.51 ≈ 0.5 预判死; 降为 B 级可选验证 (浅头诊断已无信号), 不占 S 算力** |
| A | P5-04 MLPLOB | P5-02 backbone 变体 (MLP 吃序列) | 与 Conv1D 对比, 同门禁 | P5-02 确认表示有效后 |
| A | P5-05 冻结融合 | v7 + market alpha 家族; **41-50 选权重, 51-70 完全冻结** | 4.14 纪律 + 提交前 30 秒审计 | **gated alpha (α(x) 按活动度分层) 为可选形态, 预注册后再测, 不临时加** |
| B | P5-02A-lite geometry | microprice/relative spread/L1-L2 imb/depth slope 手工家族 | 小规模 corr 诊断 | 仅在序列路线门禁失败时升级 |

## 5. target 分解 probe (GPT 二轮, 采纳为诊断)

验证 `I(X_m; sign(Y)) > I(X_m; |Y|)`: 同一 frozen encoder, 换头分别预测 sign(y) 分类 / |y| 回归, 报告各自 corr/AUC。
- 若成立 → 机制解释 MSE 失败 cosine 成功; 后续可研究 direction head + magnitude head 组合
- 成本低 (一次训练多探头), 并入 P5-02I 报告

## 6. 停止清单 (v1.4.0 继承, 不变)

TCN 大规模调参 / vanilla Transformer / 152 特征 cosine 花式变体 / 单纯 RealMLP 参数搜索 / CatBoost/XGB/MLP 换皮 / calibration 小修小补 / 同一批特征 ensemble 套娃 / 34 聚合 market 特征重做 / 月漂移后处理 (P4-05 死)。

## 7. 待用户拍板

1. **P5-02I 是否插入 P5-02 之前** (GPT 建议: 是; 成本 ~1-2h 本地, 结果直接决定 P5-02 backbone)
2. **修正 A 接受否**: M6 (event-time) 标记 SKIP (P5-01 构建已均匀化, 无差异可测)
3. **修正 B 接受否**: M7 (跨源错位) 移到 P5-02B 作为其审计门禁
4. **target 分解 probe 并入 P5-02I** 与否
5. 确认后是否开始执行 (P5-02I 一次性跑完 M0-M5)
