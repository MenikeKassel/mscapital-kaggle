# MSCapital Plan v1.9.0 — research.md 四方方案全集收敛 (GPT1 + Grok1 + deepseek + GPT2)

> 日期: 2026-08-14 | 版本: v1.9.0 (上一版 v1.8.0)
> 来源: 用户提供 `docs/_archive/reports/research.md` — 四方 AI 方案全集 (GPT1 终版 SCOPE-CI/MAGNET/REVCOH + Grok1 MagGate-Innov/SyncScat/FiLM + deepseek SCEIN/CSM-Net/TRIS-Net + GPT2 MAG-MoE/SCFI/RICS)
> 状态: **讨论定稿阶段, 未获执行许可** — 等用户拍板

## 版本历史
- v1.9.0 (2026-08-14): 吸收 research.md **全集** — 四方外部方案与本地 breakthrough-top3 收敛到同一三方向; **Grok1 + deepseek 首次入库** (v1.8.0 只吸收了 GPT1 四论 + GPT2 一论, 全仓 grep 确认 Grok/deepseek/TRIS/SyncScat 零命中); 新增 5 项 (见 §2); 审计表新增 3 处外部方案错误纠正 (见 §1)
- v1.8.0 (2026-08-14): 吸收 GPT1 四论 + GPT2 一论 — 三方案独立收敛到同一核心 ("条件化/创新信息"); 队列合并为 P5-02M → B-lite v2 → P6-04 渐进; P6-01 降级。**已归档 `_archive/plans/`**

## 0. 全集收敛分析 (本版核心)

### 0.1 四方外部方案 + 本地报告 = 同一三方向 (5/5 收敛)

| 方向 | GPT1 (终版) | Grok1 | deepseek | GPT2 | 本地 breakthrough-top3 | 判定 |
|---|---|---|---|---|---|---|
| **幅度门控 sign×mag** | MAGNET (4-bin gate) | MagGate (soft-gate) | CSM-Net (log|y| 头 + 条件符号) | MAG-MoE (quantile gate) | **COC** (定理 2/5) | 完全同构 |
| **条件创新 observed−E[·\|state]** | SCOPE-CI (z_O, z_T 两次条件化) | CondInnov-FiLM | SCEIN (μ,σ 条件预测) | SCFI (补偿器创新) | **ETCI** (事件时版) | 完全同构 |
| **反演不变跨通道相位/形状** | REVCOH (交叉谱相干) | SyncScat (散射/Gram) | TRIS-Net (shapelet min(fwd,rev)) | RICS (相干+lag-cov) | **SPCE** (相干/PLV) | 同构; deepseek 的 shapelet 是**互补机制** (见 §2.4) |

**共同核心** (v1.8.0 已确立, 本版强化): "实际 − 给定状态/上下文的期望"。五份独立方案 (四方外部 AI + 本地) 收敛 = 该方向值得认真对待, 但**实现层次必须渐进** (融合层零成本 → 特征层半天 → 表示层贵)。

### 0.2 优先级共识 (5/5)

1. **幅度门控最先跑** — GPT1/GPT2/Grok1 均明确要求先跑 4-bin/soft-gate probe (OOF 上零训练); deepseek 主张 innovation 先, 但四方多数 + 本地证据 (D1-D3、|y| corr 0.43 vs 0.156、内积 top20% 74.4%) 压倒 → **P5-02M 保持 S 级第一** (30min, 零新训练)
2. **条件创新用线性简化先行** — 四方均同意 z-score/表格版 probe 先于条件神经网络 (B-lite v2 本体); GPT2 的 SCFI 提供**更严格的对照组设计** (见 §2.2)
3. **相位/形状线先跑确定性 probe** — 全部不直接上散射/大网络; GPT2 的四臂消融 + GPT1 的确定性 probe 合并为统一 SPCE probe 设计 (见 §2.3)
4. **不再跑 Transformer/TCN/长序列** — 5/5 一致, 与停止清单兼容

### 0.3 关键分歧与裁定

| 分歧 | 外部说法 | 裁定 (本地证据) |
|---|---|---|
| 余弦最优预测形式 | Grok1: "最优 = E[y/\|\|y\|\| \| x] 方向" | **全局 cosine (本赛指标) 下 p\* ∝ E[y\|x]** (breakthrough-top3 定理 1-2 已证, 大 N 下 \|Y\| 集中; 74.4% 内积分解只对全局指标成立)。Grok1 说法仅对**逐样本** cosine 成立 — 本赛不是。不影响其架构结论 (cosine loss 训练无害) |
| 执行顺序 | deepseek: innovation 先跑, 幅度后置 | 幅度门控先 (30min 零训练 + 三方多数 + 本地证据); innovation probe 紧随 (半天) |
| plain scattering | GPT2/GPT1/本地反对 (modulus 杀相位); deepseek 用 shapelet 替代 | **谱 = cross-channel coherence (Re/Im 交叉谱, 保留相对相位), 不用 modulus 散射**; shapelet min(fwd,rev) 作为独立廉价臂并行验证 (§2.4) |
| 幅度 gate 形态 | GPT1/GPT2: 4-bin 非参数; 本地 P5-02M: 幂族 γ 网格 | **不冲突: 4-bin 作为 P5-02M 的附加臂** (§2.1), 非参数 vs 参数族对照 |

## 1. 论断 × 本地证据 × 判定 (research.md 关键论断审计)

### 1.1 与台账一致 (照单吸收, 不重复跑)

| research.md 论断 | 本地证据 | 判定 |
|---|---|---|
| market 最强 = 幅度 (corr 0.43-0.47), 方向弱 (0.56); v7 \|pred\| 与 \|y\| corr 仅 0.156 | P5-02I probes + diag_amplitude.py (D1-D3) | ✅ 一致, 已入库 |
| v7 大波动样本方向最准 (0.585 vs 小样本 0.37); 内积 top20% 占 74.4% | D1-D3 | ✅ 一致 |
| extreme AUC ≈ 0.78 → 可做幅度路由器 | P5-02I extreme probe (41-50: 0.784, 51-70: 0.744) | ✅ 一致 (注意漂移, 已在 COC 风险面) |
| 余弦最优解不是 sign-only, 也不是随意 mag×dir 组合 | breakthrough-top3 定理 1-5 (与 GPT1/GPT2 推导同构) | ✅ 三方独立同推导 |
| E[y\|x] = ∫ a·(2P(S=+\|A=a,x)−1)·p(a\|x) da → **简单 magnitude×v7 无数学依据, magnitude-conditioned experts 有** | 定理 2 (s⊥m\|x 才可分解); D1-D3 已证 s 依赖 m | ✅ 采纳: 乘法调制须逐样本 (异方差), 非全局缩放 |
| 条件创新 ≠ target residual (须区分 event surprise 与 y−ŷ) | P5-02I resid probe AUC 0.51 (y−β·v7 无方向信息) + P4-06A 聚合残差 +0.0000 | ✅ 强一致: 事件流残差化 ≠ 目标残差, 后者已 5 次证伪 |
| OOD: 全部用相对/rank/条件归一化, 禁绝对 regime | R2 归一化 LB 0.122→0.123 (+0.0023 folds); E01 ReVol-lite +0.0011 (gate 未过, 上界参考) | ✅ 方向一致; 量化上界 E01 已给 |
| 事件时间结构 (非均匀到达时刻) 从未被测过 | P5-02I 修正 A: 3s 均匀网格已抹平 event-time; 152 特征 = 窗口计数无到达时刻 | ✅ 强一致 — ETCI 事件时臂是真正新信息面 |
| sd-PNHP 式 state→event intensity 是条件创新的成熟数学 (补偿器 = 鞅创新) | 2604.23961 已核验; When Does Order Flow Matter (2607.09230) 已核验 | ✅ 采纳为 ETCI 完整版理论支柱 |

### 1.2 外部方案错误纠正 (证据在案, 不照单全收)

| # | 外部说法 | 问题 | 纠正 (证据) |
|---|---|---|---|
| 1 | deepseek SCEIN probe kill: "corr(Innov_O, O) < 0.3 → KILL (predictor 没学到东西)" | **阈值方向反了**: predictor 学得越好 → Innov 与 O 相关越低 → 按此标准会在条件化成功时误杀 | 正确判据 = innovation 是否有 alpha: 用 GPT2 SCFI 的 raw-vs-innovation 同维度对照 (§2.2) + ΔPSEUDO 门禁; corr(Innov,O) 仅作诊断信息 (预测器学到了什么), 不作 kill |
| 2 | Grok1 SyncScat kill: "market-only 提升 < +0.01 cosine → 杀" | **阈值离谱**: market-only 基线 corr 0.086, +0.01 = 12% 相对提升, 任何 probe 都过不了, 等于误杀所有可能 | 按本地纪律: blend Δ ≥ +0.0005~0.0007 + 正交性 (corr(new,现有market) < 0.7) 判生死 (breakthrough-top3 §3 E3) |
| 3 | GPT1 MAGNET continue: "4 参数 gate 需 Δcos ≥ +0.001" | 对 4-bin probe 过苛 (其自身预期就是 +0.001~0.002, 门禁顶在预期上沿) | 按 plan v1.8.0 门禁: 51-70 Δ > 0 **且** 月度 ≥ 13/20 才继续; +0.001 作为完整版 (COC 双侧调制 + log-\|y\| 头) 的目标, 不是 4-bin 的门禁 |
| 4 | deepseek SCEIN: 生产 loss = cosine + λ₁·MSE(μ_O,O) + λ₂·MSE(μ_T,T) | **multitask 陷阱重演** (4.14: lambda_cos=0.01 名义混合实际单任务) | 辅助 loss 只准在**受控消融**中出现 (P4-08A 协议), 生产 loss 禁混辅助任务 — 见 §2.5 新增停止项 |
| 5 | GPT1 SCOPE-CI: "cross-fitting nuisance model" 列为必须 | 方向正确但未说清泄漏面 | 采纳并补全: nuisance (M→O, M+O→T) 必须在**可见历史内**拟合 (past-only), OOF 生成, 与 P6R 的泄漏测试清单同规 (docs/p6r_preregistration.md) |

### 1.3 已覆盖项 (v1.8.0 / breakthrough-top3 已有, 不重复吸收)

- 条件创新三形态 (SCOPE-CI/SCFI/SCEIN/CondInnov) → B-lite v2 + ETCI 事件时臂已覆盖
- 幅度双头/乘法调制 → P5-02M + COC 已覆盖
- FiLM 融合层 (替代 concat) → v1.8.0 A 级 (条件模型完整版) 已覆盖
- soft MoE / hard regime experts → P6-04/P6-05 已覆盖
- 时间反演一致性训练 (rev augmentation) → SPCE 完整版已覆盖
- log|y| 幅度头 → COC 完整版已覆盖
- corr(new, baseline) 正交门禁 → 双轴审计 (corr<0.70) 已覆盖
- 补偿器/点过程创新 → ETCI 完整版 (条件强度臂) 已覆盖

## 2. 新增吸收项 (v1.8.0 未覆盖, 本版 5 项)

### 2.1 P5-02M 增加 4-bin 非参数门控臂 (GPT1 MAGNET + GPT2 MAG-MoE + Grok1 三方同款)
- 内容: OOF 上按 market 幅度分位 (P25/P50/P75/P90) 分 4-5 bin, 每 bin 拟合标量 w_k: ŷ_i = w_bin(m_i)·v7_i (4 参数); 与现有 γ 幂族臂对照 (参数 vs 非参数)
- 价值: 零训练, 直接验证"幅度 → 逐样本权重"是否在冻结期 (51-70) 有净增益; 三方独立设计同构 = 值得测
- 门禁: 同 P5-02M (51-70 Δ > 0 且 ≥13/20 月); 检查 gate 非常数 (Var(w_k)>0, GPT2 防退化为乘法)

### 2.2 B-lite v2 增加 SCFI 对照组 (GPT2, 最干净的证伪设计)
- 内容: 同维度 raw [O,T] vs innovation [z_O,z_T] 各训一个 RealMLP, 比较:
  - corr(pred_innovation, v7) < corr(pred_raw, v7) — 条件化应产生更正交的预测
  - blend 增益: innovation 版 ≥ raw 版
- 价值: 直接回答"条件化有没有释放新信息" (raw 是天然对照, 同特征维度同容量); 若 innovation ≤ raw → 条件化假说证伪, 条件模型永不启动 (继承 GPT2 kill: corr(pred_raw,pred_innov)>0.95 且 Δblend<0.0004 → 彻底关闭 Conditional Innovation 主线)
- 注: 同时保留 v1.8.0 的 152 双轴审计 (corr<0.70 且 ΔPSEUDO≥+0.0015) — 两套对照互补 (一个证伪条件化, 一个证伪信息增量)

### 2.3 SPCE probe 升级为四臂消融 (GPT2 RICS 设计) + GPT1 确定性特征
- 内容 (market last 10 步 = 30s):
  - A: raw last-10 flatten (时域基线)
  - B: mean/std/trend 统计
  - C: 协方差 + 对称 lag-cov (Γ_k+Γ_kᵀ, k=±1..±3)
  - D: C + cross-channel coherence (Re(C_ij), |C_ij|, 相位差 cosine)
  - D 额外跑 D(reverse) 一致性: pred_D(X) ≈ pred_D(rev(X)) (时间反演对称的架构内检验)
- 门禁: market-only corr ≥ 0.095 (基线 0.086) 且 blend Δ ≥ +0.0006; corr(新 pred, 现有 market pred) > 0.9 → 杀
- 价值: 四臂一次跑完直接定位"信息在哪一层" (时域统计 vs 二阶结构 vs 谱相干), 比 breakthrough-top3 原设计 (FFT log-power 8 频带) 更省更直接; FFT 频带臂可并入 D 臂

### 2.4 TRIS random-shapelet 探测臂 (deepseek, 与 SPCE 并行)
- 内容: 随机初始化 K 个 shapelet (L∈{3,5,7,10} × 通道), 距离 = min(dist_fwd, dist_rev) (时间反演不变), soft-min 池化 → 小 MLP; 对照 = 同结构随机特征
- 门禁 (deepseek 原设, 校准到本地量纲): random shapelet 特征集 cosine < 0.02 → 杀 shape 线 (shape matching 无信息); 通过 → 端到端 shapelet learning 才考虑
- 价值: 相位/形状线的**第二种独立表示** (谱 vs 模板匹配); 与 SPCE 共享"反演不变 + 短窗 + 跨通道"先验但机制正交 → 若 SPCE 有信号而 TRIS 无, 信息在频域; 反之在时域形态; 都无 → 相位线整体证伪
- 成本: 2-4h, 与 SPCE 并行

### 2.5 辅助 loss 生产化禁止 (deepseek SCEIN 设计触发, 修正 4 延伸)
- 条件模型完整版 (SCEIN 式 μ,σ 条件预测) 的辅助 loss (MSE(μ_O,O)+MSE(μ_T,T)) 只准出现在**受控消融** (P4-08A 协议: 同输入/同架构/同切分, 一次一个任务) — 生产 loss 禁混
- 理由: 4.14 实锤 lambda_cos=0.01 名义混合实际单任务; deepseek 的 λ₁/λ₂ 加权无消融数据支撑

## 3. 队列 v1.9.0 (增量合并 v1.8.0 + §2)

| 优先级 | 实验 | 来源 | 内容 | 门禁 |
|---|---|---|---|---|
| **S** | **P5-02M 幅度调制** (+4-bin 臂) | 本地 D1-D3 + 三方外部 (MAGNET/MagGate/MAG-MoE) | 重训 P_mag 存 m_i (5min) → pred' = v7×(m_i/median)^γ, γ 网格 + 4-bin 非参数臂; 51-70 冻结 | 51-70 Δ > 0 且 ≥13/20 月; Var(w_k)>0 防退化 | 30min; 零新架构 |
| **S** | **P5-02B-lite v2 跨源审计** (+SCFI 对照) | GPT1 四论 + GPT2 一论 + SCFI 设计 | Surprise 家族 z-score + Structural 家族 + O×T; **raw vs innovation 同维度对照**; 双 target (y 残差 + \|y\|) | 双轴 (corr<0.70, ΔPSEUDO≥+0.0015) + SCFI 对照 (innovation 更正交且 blend 更优) | 半天 |
| **S** | **P6-04 hard regime experts** | GPT1 四论 (P5-02I 分层证据) | market 状态 (活动度/波动) 分层 3-5 个 regime 各自 residual learner | f_regime1 ≠ f_regime2 (同输入不同映射增益) | hard 先行, soft MoE 后置 |
| **S** | **SPCE 四臂 probe** (+TRIS random 臂并行) | GPT2 RICS + GPT1 REVCOH + deepseek TRIS | A raw / B stats / C cov+lag-sym / D +coherence; D(rev) 一致性; TRIS random-shapelet 并行 | market corr ≥ 0.095 且 blend Δ ≥ +0.0006; corr(new,现有)<0.9; TRIS 随机基线 ≥ 0.02 | 2-4h 各, 可并行 |
| A | P6-05 soft MoE | GPT1 四论 | learned gate Σ p_k(M)·r_k | P6-04 通过后 | — |
| A | 条件创新条件模型 (SCEIN/SCFI 完整版, FiLM 融合) | GPT2 + deepseek + GPT1 | U_O = O−E[O\|M] 等 (神经网络); **辅助 loss 禁入生产** (§2.5) | B-lite v2 z-score 版 + SCFI 对照通过后 | 工程量大 |
| A | ETCI 事件时臂 (条件强度 λ(t\|state)) | 本地 ETCI + GPT2 SCFI + sd-PNHP | 原始 order/tx 到达时刻特征 (5/15/30s 事件时窗) + log-linear 强度残差 | (b) 臂 corr<0.70 且 Δ≥+0.0005; (c) 条件 vs 无条件无差 → 证伪 | 2-6h, 可与 B-lite 合并一轮 |
| B | P6-01 residual target | GPT1 四论 | 冻结 v7 OOF, 全输入 → y−ŷ_152 | 同 P5-03 (resid probe AUC 0.51 预判死) | 保留仅作可选验证 |
| B | U_152 (152 对 market 条件化) | GPT2 | Z_152 − E(Z_152\|M) | lite 通过后 | 工程量大 |
| B | SSL 预训练 | GPT1+GPT2 共同 | 掩码/预测自监督 → fine-tune | 第二阶段 | 两方案都同意放后 |
| 可选 | **P6R-01 终裁实验** (遗留) | 本地 P6R | Local vs Global Ridge, ~1.5h, 关闭 Conditional Alpha 问题最后一块拼图 | 无门禁, 科学闭合 | 上轮遗留, 待拍板 |

## 4. 停止清单 (继承 v1.8.0 + 新增)

继承全部。**新增**:
1. plain scattering / modulus 散射直接上生产 (modulus 杀相位, 与 P5-02I phase 破坏 96% 实锤冲突) — 谱线只走 cross-channel coherence
2. 条件模型生产 loss 混辅助任务 (MSE(μ,O) 等) — 必须受控消融 (§2.5)
3. shapelet 不经 random 基线直接端到端训练
4. 4-bin/soft-gate 结果不查 Var(gate) 就宣布有效 (防退化为全局乘法, GPT2 门禁)

## 5. 待用户拍板

1. 全集收敛判定接受否: 四方外部 + 本地 = 三方向 5/5 同构, 优先级幅度门控 → 条件创新 → 相位/形状 (probe 先行)
2. 新增 5 项接受否: ①4-bin 臂 ②SCFI raw-vs-innovation 对照 ③SPCE 四臂消融 ④TRIS random-shapelet 臂 ⑤辅助 loss 生产化禁止
3. 3 处外部方案纠正接受否 (§1.2: deepseek corr 阈值反了 / Grok1 +0.01 阈值离谱 / GPT1 +0.001 门禁过苛)
4. **P6R-01 终裁实验是否先跑** (上轮遗留, 1.5h, 科学闭合 H2)
5. **是否开始执行 P5-02M** (30min 零训练, 含 4-bin 臂) — 全部方案一致的第一优先

## 附: research.md 与现有文档的关系

- research.md (docs/_archive/reports/) = 四方外部方案原文存档, 保留不动
- breakthrough-top3-2026-08-14.md (research/) = 本地对应研究 (定理 1-5 + COC/ETCI/SPCE 候选卡 + kill criteria), 为 v1.9.0 的详细展开
- 本 plan = 权威路线 (唯一 current source of truth, 继承 9861dc6 权威链)
