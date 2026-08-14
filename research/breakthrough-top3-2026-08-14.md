# MSCapital 突破研究: 余弦最优预测推导 + Top 3 突破候选 (2026-08-14)

> 状态: 研究定稿版 v0.9 (文献表/比赛表待子代理一手核验后补 v1.0)
> 输入: 用户研究任务书 + plan v1.8.0 + P5-02I 五条实锤 + D1-D3 幅度诊断 + E01 ReVol-lite 结果
> 纪律: 新信息 > 新表示 > 新目标 > 新模型 > 调参; 每个候选回答"用了哪个以前没用的随机变量"; 全部候选配 PSEUDO 门禁 + kill criterion

---

## 0. Executive Conclusion

如果只有 3 次大实验机会，跑这三个（按优先级）：

1. **COC (Cosine-Optimal Calibration, P5-02M 的完整版)** — 把预测改成 `p = m̂ × d̂` 的度量最优形态：用 market 幅度模型做**双侧**调制（尾部放大 + 主体衰减/翻转），幅度模型用 log|y| 目标重训。零新架构，30 分钟出第一个冻结验证。
2. **ETCI (Event-Time Conditional Innovation)** — 在**事件时间**（非均匀）上做状态条件化创新：`U = observed − E[·|market state]`，用条件强度（state-dependent Hawkes-lite）+ FiLM 融合替代 concat。这是计划队列 B-lite v2 之后真正未测过的信息：事件时间结构在 3s 均匀网格构建时已被抹掉（P5-02I 修正 A），至今无人测过。
3. **SPCE (Spectral Phase-Coherence Encoder, 反直觉)** — 短窗 (≤30s) 跨通道相干性 + 相位特征 + 线性头；升级版 = 散射变换。唯一直接攻击"相位形态"（surrogate 中最致命的一刀，phase 破坏损失 96%）且理论上抗漂移（尺度不变 + 形变稳定）的表示。

核心洞察一句话：**余弦指标的最优预测就是 E[y|x]，而 E[y|x] = E[sign|x]·E[|y||x]（条件独立时）——所以"幅度调制"不是锦上添花的 trick，它就是度量最优解的数学形式；你们已经发现了它（P5-02M）但只做了放大半边，D1-D3 数据还支持翻转/衰减半边。**

---

## 1. 余弦指标下最优预测的严格推导 (本次新增的核心理论)

### 1.1 设定

测试集 N 个样本，特征 $x_i$，未知目标 $y_i$，预测 $p_i = p(x_i)$。指标（全局 uncentered cosine，与你们 74.4% 内积分解分析一致）：

$$S(p, y) = \frac{\langle p, y \rangle}{\|p\| \cdot \|y\|} = \frac{\sum_i p_i y_i}{\sqrt{\sum_i p_i^2}\sqrt{\sum_i y_i^2}}$$

### 1.2 定理 1: 最优预测 = E[y|x]（比例意义下）

令 $s(x) = \mathbb{E}[y|x]$。对任意预测函数 $p$：

$$\mathbb{E}[S] \le \frac{\sqrt{\mathbb{E}[s^2]}}{\sqrt{\mathbb{E}[y^2]}}, \qquad \text{等号当且仅当 } p(x) \propto s(x)$$

证明：大 N 下 $\|p\| \approx \sqrt{N\mathbb{E}[p^2]}$，$\|y\| \approx \sqrt{N\mathbb{E}[y^2]}$，$\langle p,y\rangle \approx N\mathbb{E}[p\,y] = N\mathbb{E}[p\,s]$（重期望）。于是

$$\mathbb{E}[S] \approx \frac{\mathbb{E}[p\,s]}{\sqrt{\mathbb{E}[p^2]}\sqrt{\mathbb{E}[y^2]}} \le \frac{\sqrt{\mathbb{E}[s^2]}}{\sqrt{\mathbb{E}[y^2]}}$$

（Cauchy–Schwarz），等号当且仅当 $p \propto s$。∎

**推论 1.1（全局尺度无关）**: $S(cp,y)=S(p,y)$。全局缩放不改变分数——**只有跨样本的相对尺度（形状）决定分数**。任何"把预测缩放到某个全局目标尺度"的努力都是浪费。

**推论 1.2（回答任务书问题）**: 应预测 $\mathbb{E}[y|x]$，**不是** $\mathbb{E}[y/\|y\||x]$（后者是逐样本余弦平均时才最优，那种指标下只有符号有意义、幅度全部无关；你们的 74.4% 内积分解只对全局指标成立，所以幅度必须进预测）。

### 1.3 定理 2: 符号×幅度分解

$y_i = s_i m_i$，$s_i=\mathrm{sign}(y_i)$，$m_i=|y_i|$。则

$$\mathbb{E}[y|x] = \mathbb{E}\big[m \cdot \mathbb{E}[s|x,m] \,\big|\, x\big]$$

若 $s \perp m | x$（条件独立），精确成立 $\mathbb{E}[y|x] = \mathbb{E}[m|x]\cdot\mathbb{E}[s|x]$。

**含义**: 幅度信息**只能通过乘在方向上的形式**进入最优预测。这给了 P5-02M（乘法调制）数学依据——不是启发式。两个推论：
- 幅度模型和方向模型的正确接口是**乘积**，不是加法混合（加法混合是当前 v7+market blend 的做法，它把两个不完整估计相加，等价于在错误的结构上做近似）。
- 度量对"幅度估计"本身不奖励：只有幅度×方向联合对了才得分。所以幅度模型的价值 = 把预测质量集中在 $\mathbb{E}[m|x]$ 大的样本上。

### 1.4 定理 3: 分数集中在尾部（为什么大波动样本值钱）

在最优预测 $p^*=s$ 下，样本 $i$ 对分子 $\langle p,y\rangle$ 的期望贡献为 $\mathbb{E}[s_i^2] = \mathbb{E}[\mathbb{E}[y|x_i]^2]$。重尾分布下（你们实测 top20% 样本贡献 74.4% 内积、top5% 贡献 56.5%），**分数由 $\mathbb{E}[m|x]$ 的尾部主导**。推论：
- 校准努力应按 $\hat m$ 加权（尾部样本的误差对分数的影响大得多）；
- extreme 分类器（AUC 0.78）不是花哨诊断，它就是**分数路由器的正确数学角色**；
- 尾部方向精度（v7 在最大 |y| 层 0.585）是全项目杠杆率最高的单个数字。

### 1.5 定理 4: 噪声估计 ⇒ 逐样本衰减（不是全局收缩）

若模型输出 $\hat s = s + \eta$（$\eta\perp x$），则 $\hat s$ 的函数中分数最优的是 $\mathbb{E}[s|\hat s]$（后验均值/去噪）。高斯近似：

$$p_i = \hat s_i \cdot \frac{\tau(x_i)}{\tau(x_i) + \nu(x_i)}, \qquad \tau=\text{信号方差}, \nu=\text{噪声方差}$$

**关键**: 全局收缩因子在余弦下抵消（推论 1.1），但**逐样本（异方差）衰减不抵消**——它正是"把主体压小"的数学形式。你们的 D1-D3 数据（v7 在小 |y| 层方向正确率 0.37 < 0.5）意味着那些样本的预测在**主动拖累分子**：把它们的输出压向 0（甚至翻转符号——若 0.37 是稳定反转则最优输出是 −v7），分子分母同时受益。

### 1.6 定理 5: 尾部专家混合 = 精确最优，不是 hack

令 $E$ = "极端样本"事件（如 $m > q_{90}$）。则

$$\mathbb{E}[y|x] = \mathbb{P}(E|x)\,\mathbb{E}[y|x,E] + (1-\mathbb{P}(E|x))\,\mathbb{E}[y|x,\neg E]$$

**度量最优预测精确地是一个二专家混合，权重 = 极端概率**。你们的资产齐了：$\mathbb{P}(E|x)$ = market 的 extreme 分类器（AUC 0.78），$\mathbb{E}[y|x,E]$ ≈ v7 在尾部的表现（0.585），$\mathbb{E}[y|x,\neg E]$ ≈ 0（v7 在那里 0.37，接近纯噪声偏反）。这就是"Magnitude-Gated Alpha"的严格版本：不是 learned softmax gate，是条件期望分解本身。

### 1.7 推导的实践结论（全部可直接落地）

1. 输出 $\mathbb{E}[y|x]$ 的完整幅度结构；不做 rank-only / sign-only / clip。
2. 幅度与方向的接口 = 乘积（$\hat m \cdot \hat d$）。
3. 逐样本衰减：低 $\hat m$ 样本压向 0；测试翻转（若 0.37 稳定，翻转正确率 0.63 > 0.5）。
4. 尾部专家 + extreme 路由器 = 定理 5 的直接实现。
5. 幅度模型训练目标用 **log|y|**（尺度不变）+ 相对特征（抗漂移）；全局尺度永远不用校准。
6. 幅度模型的相对结构是唯一需要跨期稳定的东西——test/valid scale 门禁检查的正是这个。

---

## 2. Top 3 候选架构

### 候选 1: COC — Cosine-Optimal Calibration（余弦最优校准）【研究代号: COC-00】

| 项 | 内容 |
|---|---|
| **来源** | 本次定理 1-5 推导（新）；P5-02M 为计划队列中的最小实例；幅度双头已在 P5-02I probe 中有实证基础 |
| **为什么匹配** | ① 幅度富矿 (market |y| corr 0.43-0.47 vs v7 0.156) 是唯一已知且未变现的信息差；② D1-D3: v7 尾部方向 0.585 最强——正好在余弦最看重的样本上；③ 余弦 74.4% 集中在 top20%——尾部调制杠杆最大；④ 全局尺度无关性使 log|y| 训练天然抗漂移 |
| **利用了哪个新随机变量** | 没有新 R.V.，但有一个**新目标**：I(X; |y|) 通过专用 log-|y| 目标显式建模（现在 |y| 只以 v7 输出幅度的副产品形式存在，corr 仅 0.156）。按纪律排序: 新目标 > 新模型 |
| **为什么旧方法没覆盖** | 队列中的 P5-02M 只做放大半边 `v7×(m/median)^γ`；乘法双头 (P5-02) 是启发式乘积；本候选补上: ① 数学依据（定理 2/5）；② **双侧**调制（尾部放大 + 主体衰减/翻转——0.37 反预测主体 25.6% 的内积质量）；③ 尾部专家混合（定理 5，AUC 0.78 路由器）；④ log-|y| 尾部加权幅度头。不是 concat、不是 residual、不是新架构——是**目标层**的修正 |
| **预期新增信息** | I(X; \|y\|) 的显式变现 + P(sign, \|y\|) 联合结构 |
| **最大风险 (OOD)** | ① extreme AUC 已从 41-50 的 0.784 掉到 51-70 的 0.744——幅度模型本身在漂移，test 期可能更弱；② 0.37 反预测可能是噪声（月度不稳定则翻转无益）；③ 若 test 期 |y| 分布的相对结构改变（不只尺度），调制权重会系统性错位。缓解: 全部用相对特征 + log 目标 + 月度一致性门禁 |
| **最小验证 (probe)** | 完全零训练起步: 用已有 v7 OOF + market OOF 预测，网格试 `p' = v7·(|p_market|/median)^γ` (γ∈{0.2..2}) + 主体衰减 `p' = v7·min(1,(m/median)^γ)` + 底部 10% 翻转，51-70 冻结测 cosine。30min。若 |p_market| 不够，加 2h 训练 log-|y| 幅度头 (market 序列, cosh loss, 尾部加权) |
| **Kill criterion** | ① 51-70 Δ < +0.0003 → 杀（幅度调制在冻结期无净增益）；② 月度正率 < 13/20 → 杀；③ 幅度模型 test/valid 相对结构漂移 (rank corr of m̂ 跨期 < 0.8) → 杀;④ 尾部 (top10% m̂) 方向正确率在 51-70 跌破 0.52 → 整个前提失效, 杀 |
| **成功后的完整版** | ① log-|y| 幅度头 (market 特征, 尾部加权 loss) 生产化; ② 尾部方向专家: 在 m̂ top 10-20% 子集上重训/微调方向模型 (加权 cosine loss); ③ 定理 5 二专家混合: `p = p̂(E)·dir_tail + (1−p̂(E))·dir_bulk`; ④ 与现有 blend 家族做正交性检查后融合; ⑤ 全流程走 Protocol-v2 |

### 候选 2: ETCI — Event-Time Conditional Innovation（事件时条件创新）【研究代号: ETCI-00】

| 项 | 内容 |
|---|---|
| **来源** | 计划 v1.8.0 已核验: State-dependent Hawkes (2604.23961, 2026-04)、When Does Order Flow Matter (2607.09230, 2026-07)、Cont 2010 冲击、Evans-Lyons 2002 订单流惊奇、Hasbrouck 1991 VAR 创新; 点过程补偿器 = "观测−期望"的严格数学形式 (鞅创新) |
| **为什么匹配** | ① 三方案独立收敛到条件化/创新信息 (GPT1+GPT2+本地证据) = 方向值得认真对待; ② P5-02I: low-activity 增益 lo +0.0021 > hi +0.0006 = regime 条件化有实证; ③ **事件时间从未被测过**: P5-02I 修正 A 已承认 3s 均匀网格抹掉了非均匀事件时间结构, order/tx 流的到达时刻、间隔、突发性在现有 152 特征和全部序列模型中均未表达 = 真正的新信息; ④ 补偿器创新残差是平稳过程 (条件期望吸收 regime 水平) = 结构性抗漂移 |
| **利用了哪个新随机变量** | I(Y; O\|M) 与 I(Y; T\|M,O): 状态条件下的**事件到达时刻** (duration/gap/burstiness) + 条件强度残差 U = observed − ∫λ̂。事件时间戳此前从未作为信息使用 (序列是均匀网格, 特征是无条件窗口计数) |
| **为什么旧方法没覆盖** | ① B-lite v2 (队列) 测的是**均匀网格上的 z-score surprise**——z-score 是条件创新的线性简化, 但事件时间结构在网格上不存在, 必须回到原始快照秒数; ② M01 是无条件窗口计数, 无"给定状态的条件期望"; ③ P3-02 是 concat 融合 (已失败 +0.00095), ETCI 用 FiLM 条件化 (scale/shift 注入) 而非 concat——不同机制; ④ 不是 Hawkes 换皮: 用的是**短记忆 (≤30s) 条件强度**而非长程自激核 (与"无长程依赖"实锤一致) |
| **预期新增信息** | I(Y; O\|M) + I(Y; T\|M,O) 的事件时间分量; 若为 0 → 该方向证伪, 省下条件模型全部工程量 |
| **最大风险 (OOD)** | ① 条件强度模型本身过拟合训练期 regime → 残差变成噪声; 缓解: log-linear 小容量 + 折内拟合 + past-only; ② 事件到达率在 test 期可能系统性变化 (月度漂移) → 条件期望吸收大部分, 残差标准化后应平稳, 需门禁检查; ③ 订单流条件预测本身噪声大 (plan v1.8.0 已标注) |
| **最小验证 (probe)** | 三臂一次跑: (a) z-score surprise 臂 (B-lite v2 本体, 半天); (b) **事件时间臂**: 从原始 order/tx 事件流构建 duration/间隔/突发特征 (5/15/30s 事件时间窗), 加进 152+market 特征集, 双轴审计 (corr<0.70 且 ΔPSEUDO≥+0.0015); (c) **条件强度臂** (仅当 b 有信号): 6-8 个 mark 的 log-linear 强度 λ_k(t|state), 残差特征。2-6h |
| **Kill criterion** | ① (b) 臂 corr(新特征,152)≥0.70 → 事件时间信息与现有特征重叠, 杀; ② (b) 臂 ΔPSEUDO<+0.0005 → 事件时间无独立信息, 杀整个 ETCI 线; ③ (c) 臂条件残差 Δ 与 (b) 无条件版无显著差 → "条件化"假说证伪 (z-score 已足够), 不上条件模型; ④ 月度正率 < 13/20 |
| **成功后的完整版** | ① state-dependent 条件强度模型 (log-linear → 小 MLP, 短核); ② FiLM 条件化融合层替代 concat (state→per-channel scale/shift); ③ 三源 encoder (market 短窗 channel-mixing + order 事件时 + tx 事件时) → 创新特征 → RealMLP cosine 头; ④ M7 对齐 vs 错配门禁 (对齐 Δ > 错配 Δ) |

### 候选 3: SPCE — Spectral Phase-Coherence Encoder（谱相位相干编码器, 反直觉）【研究代号: SPCE-00】

| 项 | 内容 |
|---|---|
| **来源** | 信号处理/神经科学: cross-spectral coherence (经典), 散射变换 (Bruna & Mallat 2013 TPAMI; **金融应用已核验: ICASSP 2019 "Maximum-entropy Scattering Models for Financial Time Series", DOI 10.1109/icassp.2019.8683734**), GAF 时序图像 (Wang & Oates 2015), 相位锁定值 PLV (神经科学), 双线性池化 (Fukui 2016) |
| **为什么匹配** | 逐条对 P5-02I 五条实锤: ① 短程 ≤30s → 短窗 (10-30 步) FFT, 可分辨周期 6-300s, 恰覆盖目标频带; ② 跨通道同步是核心 (desync 损失 77-94%) → **相干性 = 频域里的跨通道同步**, 这是该结构的原生数学语言; ③ 非线性相位形态 (phase 破坏损失 96%, 最致命一刀) → 交叉谱相位 + 高阶散射系数 (模-非线性级联捕捉包络交互), 现有 152 特征全部是时域统计, 无任何谱/相位特征 (已 grep 确认); ④ 无时间箭头 → 相干幅度与 PLV 天然时间反演对称 (交叉谱取共轭, |·| 不变); ⑤ 抗漂移: 相干性尺度不变 + 局部窗 + 散射的形变稳定性 (Lipschitz 于时间翘曲 = 对"交易节奏变化"这类漂移的理论鲁棒性, 这是 Transformer/TCN 没有的性质) |
| **利用了哪个新随机变量** | 无新 R.V.——是**新表示** (X_m 的跨通道联合谱结构: 二阶互统计量)。诚实标注: 按纪律排第三位; 但它攻击的是 surrogate 中信息密度最高的单一结构 (相位/同步), 且唯一带抗漂移理论 |
| **为什么旧方法没覆盖** | ① 152 特征 = 时域统计 (窗口/EWM/盘口几何/事件流), 无 FFT/相干/相位 (已 grep 全仓确认); ② MultiRocket (scout E06) 是随机卷积, 机制不同 (时域 motif 而非频域); ③ 2.5D grid 失败 = 原始 通道×时间 图像 + CNN, 无显式频域/相干结构; ④ Path Signature 是路径几何积分, 非谱; ⑤ 不是 Transformer/TCN 换皮——没有可训练时序核, 是**固定变换 + 轻量头** |
| **预期新增信息** | X_m 的跨通道联合二阶谱结构 (相干/相位对齐) 对 Y 的条件信息; 若 surrogate 的"跨通道同步+相位"判断正确, 这是最直接的特征化 |
| **最大风险 (OOD)** | ① 30s 窗内可分辨周期 6-30s 只有 1-5 个周期, 谱估计方差大 → 特征噪声大; ② 相位结构可能 regime 特异 (test 期相位形态变化); ③ 3s 均匀网格 (carry-forward) 已损失亚秒级结构, 谱上限被网格限制。缓解: 多窗平均 + 相干幅度 (去相位噪声) + 线性头低容量 |
| **最小验证 (probe)** | numpy FFT 特征 (零依赖): 3 个短窗 (10/20/30 步) × 关键通道 (mid, spread, L1/L2 depth, vol, OFI, imbalance) → 每窗每通道 log-power 8 频带 + 通道对相干幅度 + PLV; ~200-400 维 → **线性头** cosine loss, 与"同特征集时域统计 + 线性头"对照; 也测 corr(特征, 152) 双轴。2-4h, 不动大型模型 |
| **Kill criterion** | ① 谱特征线性头 ≤ 时域统计线性头 (valid 与 PSEUDO 双判) → 谱表示无增量, 杀; ② corr(谱特征, 152) > 0.85 → 信息重叠 (不太可能但需测); ③ PSEUDO Δ < +0.0005; ④ 月度一致性 < 13/20 |
| **成功后的完整版** | ① 多通道 1D 散射变换 (kymatio, 2 尺度 ≤30s) + log 池化 → 固定表示 → RealMLP/ridge; ② 与 COC 幅度头结合 (谱特征做 m̂ 的备选源); ③ 时间反演一致性: 市场编码器训练加 rev(x) 增强 + f(x)≈f(rev x) 一致性 loss (标签保持增强, 直接编码"无时间箭头"先验, 方差降低零偏差); ④ 融合进现有 blend |

---

## 3. 实验序列 (最值得先跑的)

### Experiment 1: COC 零训练 probe 【30min-2h】
- **Hypothesis**: `p = v7 × w(m̂)` 在余弦度量下优于当前 v7/market blend, 因为度量最优解 = E[sign|x]·E[|y||x], 而 m̂=market 幅度 (corr 0.43) 远好于 |v7| (corr 0.156)
- **Implementation**: 复用 output/p5_01_market_sequence (market OOF) + v7 OOF; 网格: γ∈{0.1,0.25,0.5,1,2}, 三种 w: 放大 `(m/med)^γ`, 双侧 `min(1,(m/med)^γ)`(只衰减), 翻转 (底部 10% 翻转符号); 51-70 冻结; 每月 cosine
- **Baseline**: 当前生产 blend 0.14255 (PSEUDO 0.14255 协议下)
- **Metric**: 51-70 cosine Δ vs baseline, 月度正率, corr(p', v7), test/valid 尺度
- **Expected**: 若 D1-D3 模式在 51-70 稳健, 放大臂 +0.0005~+0.002; 双侧臂可能更多 (主体 25.6% 内积质量)
- **Kill criterion**: Δ<+0.0003 或 月度<13/20 或 尾部正确率<0.52 → 杀, 不做完整版
- **Next step**: 通过 → 训练 log-|y| 幅度头 + 尾部专家 (完整版 COC)

### Experiment 2: ETCI 三臂 probe 【2-6h】
- **Hypothesis**: I(Y;O|M)+I(Y;T|M,O) 的事件时间分量 > 0, 且条件化 (相对状态) 优于无条件
- **Implementation**: (a) B-lite v2 z-score surprise 臂 (计划队列本体, 半天); (b) 事件时间臂: 原始 order/tx 流 → duration/gap/burst 特征 (5/15/30s 事件时间窗) + 152 + market 聚合 → 双轴审计; (c) 条件强度臂: 6-8 mark log-linear λ(t|state), 残差特征 (仅当 b 通过)
- **Baseline**: 152+market 特征集 (corr/ΔPSEUDO 双轴 vs 现有特征)
- **Metric**: corr(新特征,152)<0.70 且 ΔPSEUDO≥+0.0015; 月度一致性
- **Expected**: z-score 臂预计 +0.0005~+0.0015 (E01 ReVol-lite 同类归一化只到 +0.0011); 事件时间臂未知, 是真正的新信息
- **Kill criterion**: (b) corr≥0.70 或 Δ<+0.0005 → 杀整线; (c) 条件 vs 无条件无差 → "条件化"证伪, 条件模型永不启动
- **Next step**: 通过 → FiLM 三源编码器 (完整版 ETCI)

### Experiment 3: SPCE probe 【2-4h】
- **Hypothesis**: 短窗跨通道相干/相位特征携带 152 特征与市场编码器未表达的联合谱信息, 且相干性 (时间反演对称 + 尺度不变) 抗漂移
- **Implementation**: numpy FFT → 3 窗 × 8 通道 × 8 频带 log-power + 6-8 通道对相干 + PLV ≈ 200-400 维 → 线性头 cosine loss; 对照 = 同通道时域统计 + 线性头
- **Baseline**: 时域统计线性头 + 现有生产 blend
- **Metric**: valid/PSEUDO cosine Δ, corr(谱,152), 月度
- **Expected**: 中性偏正; 若相位/同步结构真如 surrogate 所示, 线性头即可见增量; 若谱噪声主导, 立即死
- **Kill criterion**: 谱 ≤ 时域 (双 fold) → 杀; Δ<+0.0005 → 杀
- **Next step**: 通过 → 散射变换完整版 + 时间反演一致性训练

---

## 4. 对"条件化/创新信息"假说的裁定（原则 9 要求）

**结论: 方向正确但必须分层验收, 且"线性简化先行"的顺序是对的。**

支持面 (证据实锤):
- 三方案独立收敛 (GPT1/GPT2/本地) — 弱证据但值得认真对待
- P5-02I: low-activity lo +0.0021 > hi +0.0006 — regime 条件化增益的直接证据
- R2 归一化 (绝对→相对) LB +0.001, 4/4 fold +0.0023 — "相对状态"方向已变现
- When Does Order Flow Matter (2607.09230, 已核验) — "state first, flow additive and conditional" 的外部一致
- 补偿器/鞅创新的数学严谨性 — 条件创新的严格定义就是点过程补偿器

风险面 (必须门禁):
- P4-06A: 600s 状态条件化 → 残差 = 0.0000 (链条曾断裂)
- P4-01a: 演化 ratio 特征区分力 ≈ 0 (surprise 粗版无信号, 但那是解释性画像非 target 残差, 且 ratio≠z-score)
- resid probe AUC 0.51 (残差方向无信息 — 但这是"对 y−v7 残差的方向", 不等于"条件化特征对 y 无信息")
- E01 ReVol-lite: 归一化表示 PSEUDO 只到 +0.0011 (gate 未过) — **同类思想的近期定量上界**
- P3-02: concat 融合 +0.00095 gate F — 融合形态错了就是负贡献

**裁定**: 若 B-lite v2 的 z-score 臂与事件时间臂都过不了双轴门禁, 则"条件化/创新信息"假说在 MSCapital 数据上证伪, 停止条件模型/MoE 全部工程量, 资源转向 COC 完整版与 SPCE。这不是失败——是省下数周工程。

---

## 5. Top 10 相似比赛 / 数据集 (结构对比)

> 数据生成结构相似度: ★ 数量 + 是否含 market/order/trade 三件套 + 指标族 + 漂移形态。✅=本次或仓库已核验。

| # | 比赛/数据集 | Market快照 | Order事件 | Trade事件 | horizon | target | metric | 漂移 | 相似度与理由 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Optiver — Trading at the Close** (Kaggle 2023, 已核验) | ✅ 10min L1/L2 快照+盘口变化 | ✅ | ✅ | 10min→次日收盘 | log return | RMSE | 中 | ★★★★★ 三件套最齐全; 前排方案 (imbalance/urgency 特征 + NN/GBDT) 已吸收进 M05 |
| 2 | **Optiver Realized Volatility** (Kaggle 2022, 已核验) | ✅ 10min 盘口快照 | ✅ 盘口变化 | 部分 | 10min | realized vol | RMSPE | 高 | ★★★★☆ **波动率目标 = 我们的幅度富矿同构**; KNN 冠军思路已试 (M04 吸收), 未充分利用"波动率可预测→门控" |
| 3 | **G-Research Crypto Forecasting** (Kaggle 2022, 仓库已侦察) | 部分 (分钟 OHLCV) | ✗ | ✗ | 15min | residualized return + 时变权重 | **Pearson corr** | 高 (权重随漂移) | ★★★★☆ **指标族最接近** (重尾收益上的相关类目标); 前排方案用 correlation-loss 训练 LSTM, 与我们的 cosine loss 同族; 无 LOB 事件 |
| 4 | **DRW Crypto Market Prediction** (2025, 仓库已核验) | ✅ 分钟 bid/ask | ✗ | ✅ 量 | 分钟级 | return | Pearson | **极高 (未来数据评分)** | ★★★★☆ 漂移即赛题; 冠军 MLP+AE 特征 (方案已入库) |
| 5 | **FI-2010** (2017 基准, 已知) | ✅ 10 档 LOB | ✅ (状态内) | ✗ 独立流 | 10/50/100 tick | 3 类方向 | accuracy | 中 | ★★★★☆ 纯 LOB 结构; 分类目标; DeepLOB/TLOB 评测场 |
| 6 | **LOBSTER / Nasdaq TotalView-ITCH** (已知) | ✅ 可重建 | ✅ **完整消息流** (add/cancel/execute) | ✅ 执行消息 | 任意 | 任意 | 学术 | 无 | ★★★★★ **事件流结构最像 order.feather+transaction.feather**; 事件语法/Hawkes 的外部 sanity check |
| 7 | **CryptoLOB** (2024, 已知) | ✅ 10 档 | ✅ | ✅ | 多种 | mid 方向/回归 | 学术 | 中 | ★★★★☆ 加密 LOB + 事件流; 比 FI-2010 新 |
| 8 | **Jane Street Real-Time Market Data Forecasting** (2024, 仓库已核验) | ✗ 匿名特征 | ✗ | ✗ | 实时 | responder | weighted zero-mean R² | 极高 (流式) | ★★★☆☆ 数据形态远, **漂移方法论最先进** (recent/static 分治, 流式无前视) |
| 9 | **Numerai** (持续, 已知) | ✗ 匿名特征 | ✗ | ✗ | 回合制 | rank signal | **rank IC** | 极高 (era 漂移) | ★★★☆☆ 目标族 (相关类) + 漂移对抗文化; 大量"correlation objective"实战经验 |
| 10 | **IMC Prosperity / SIG 挑战** (已知) | ✅ 模拟 LOB | ✅ | ✅ | 秒级 | 做市/交易 | PnL | 低 (模拟) | ★★★☆☆ 结构像但目标完全不同 (PnL), 参考价值在事件语法 |

**结构结论**: ① 指标族上 G-Research/Numerai 的 correlation-objective 经验与 cosine 最亲; ② 事件流结构上 LOBSTER/ITCH 与 CryptoLOB 最像; ③ Optiver RV 的"波动率可预测"与本赛幅度富矿 (0.43) 是同一现象的两种表述 — 它的前排方案如何利用波动率预测是 COC 的免费参考。**没有任何公开比赛用 uncentered cosine + 三件套 + 长漂移 → 本赛组合独特, 只能拼装迁移。**

## 6. 文献地图 (方向/论文/年份/核心思想/源码/关联)

### A. Limit Order Book 深度学习
| 论文 | 年份 | 核心思想 | 源码 | 与本赛关联 |
|---|---|---|---|---|
| DeepLOB (Zhang et al.) | 2019 | CNN+Inception+LSTM 端到端 | [github.com/zcakhaa/DeepLOB](https://github.com/zcakhaa/DeepLOB) | 本地 TCN/Transformer 负结果已覆盖其路线; 不直接迁移 |
| TLOB (Berti et al.) | 2025 (2502.15757 ✅) | 双注意力 (token+channel) 短窗 | [github.com/LeonardoBerti00/TLOB ✅](https://github.com/LeonardoBerti00/TLOB) | **通道注意力与"跨通道同步"实锤同构**; 借鉴机制不迁架构 (长注意力已死) |
| HLOB (Briola et al.) | 2024 (2405.18938 ✅, ESWA) | 信息论特征选择 + 大模型; "信息持久性" | 社区复刻 [shawsignaldev/hlob ✅](https://github.com/shawsignaldev/hlob-depth-persistence-study); 官方仓库本次未确认 | 与本赛信息论框架同构; 信息持久性结论支持短程 |
| Deep LOB Forecasting (Briola et al.) | 2024 (2403.09267 ✅) | 微观结构引导深度模型 (HLOB 前作) | — | 同上; 作者: Briola/Bartolucci/Aste |
| LOBERT | 2025 (2511.12563 ✅) | BERT 式 LOB 预训练+微调 | plan 已核验 | plan B 级后置; 本地 SAE 重叠先例提示谨慎 |
| LOB Benchmark Study (Zhang et al.) | 2023 (2308.01915 ✅) | 8 模型标准评测 (DeepLOB/Transformer/TCN...) | — | 同结构数据上的模型上限参考; 本地已覆盖 |
| Event-based LOB Simulation | 2025 (2502.17417 ✅) | 事件级 LOB 生成 | plan 已核验 | ETCI 事件 mark 语法参考 |
| TradeFM | 2026 (2602.23784 ✅) | 交易金融基础模型 | plan 已核验 | 后置 (plan B) |

### B. 事件模型 / 点过程 / 订单流
| 论文 | 年份 | 核心思想 | 源码 | 与本赛关联 |
|---|---|---|---|---|
| Cont-Kukanov-Stoikov | 2014 (JFE) | OFI → 短期收益 | — | 152 特征 OFI 家族源头; 无条件版已用尽 |
| Hasbrouck | 1991 (JoF) | 成交创新 VAR 分解: 不可预测分量=信息 | — | **ETCI 成交创新理论基础** |
| Evans & Lyons | 2002 (JPE, DOI 10.1086/324391 ✅) | 订单流惊奇驱动汇率 | — | "订单流−期望=信息"经典实证 |
| VPIN (Easley, López de Prado, O'Hara) | 2012 (RFS) | 成交量同步订单流毒性 | — | 流毒性/意外流方向参考; 预测力有争议, 仅作残差特征候选 |
| Bacry et al. | 2015 | Hawkes 金融综述 | — | 背景 |
| State-dependent Hawkes | 2026 (2604.23961 ✅) | 强度参数随市场状态变化 | plan 已核验 | **ETCI 直接理论: λ(t\|state)** |
| **When Does Order Flow Matter** (Jeon) | 2026 (2607.09230 ✅) | L2 流动性状态切换下订单流才预测收益: "state first, flow additive & conditional" | — | **最强外部锚点**: 与本赛 lo +0.0021/hi +0.0006 分层证据同构; ETCI 的主要理论支柱 |
| Neural Hawkes / THP / Intensity-free | 2016-2021 | 神经网络强度与补偿器 | 多处公开 | 完整版备选; 本地先 log-linear |

### C. 幅度/波动率门控 (COC 家族)
| 论文 | 年份 | 核心思想 | 源码 | 与本赛关联 |
|---|---|---|---|---|
| **Moreira & Muir: Volatility-Managed Portfolios** | 2017 (JF) | 按预测波动率缩放敞口, 夏普大幅提升 | — | **COC 的金融祖先**: "幅度调制"在组合层面的经典实证 — 与本赛"market 预测 |y| 0.43" 完全同构 |
| Kendall & Gal | 2017 | 异方差回归 (均值+方差双头) | — | 定理 4 的 NN 实现 |
| Bishop MDN | 1994 | 混合密度网络 | — | 分布建模备选 |
| IQN (Dabney) | 2018 | 隐式分位数回归 | — | 尾部建模备选 |
| 本地 extreme probe | 2026 | |y|>p90 AUC 0.78 | scripts/diag_amplitude.py | 路由器 (定理 5) |

### D. 条件表示 (ETCI 融合层)
| 论文 | 年份 | 核心思想 | 源码 | 与本赛关联 |
|---|---|---|---|---|
| FiLM (Perez et al.) | 2018 (ICLR) | 条件 scale/shift 注入 | 公开 | ETCI 融合层 (替代已失败的 concat) |
| Reconditionor/SOLID | 2023 (2310.14838 ✅) | 残差-上下文依赖检测+局部适配 | [github.com/HALF111/calibration_CDS ✅](https://github.com/HALF111/calibration_CDS) | scout E02 已列; 诊断先行纪律 |
| Hypernetworks (Ha) | 2016 | 权重生成 | — | 备选 |

### E. 谱/形状/时间反演 (SPCE 家族, 反直觉)
| 论文 | 年份 | 核心思想 | 源码 | 与本赛关联 |
|---|---|---|---|---|
| **Invariant Scattering Networks** (Bruna & Mallat) | 2013 (TPAMI) | 小波散射: **形变稳定** + 平移不变 | [github.com/kymatio/kymatio](https://github.com/kymatio/kymatio) | SPCE 升级版; **形变稳定性 = 抗"交易节奏漂移"的理论依据** (Transformer/TCN 没有的性质) |
| **Max-Ent Scattering for Financial TS** | 2019 (ICASSP ✅) | 散射系数用于金融时序建模 | — | **金融散射直接先例** (本次 OpenAlex 核验) |
| GAF (Wang & Oates) | 2015 | Gramian 角场→图像+CNN | — | 相位/角关系显式化 |
| Bilinear pooling (Fukui) | 2016 | 外积跨通道二阶交互 | — | 通道同步的显式二阶建模 |
| T-LBaF (Passalis) | 2019 | 时序 bag-of-features 局部形状 | — | 局部形状摘要低成本先例 |
| ROCKET/MiniRocket/MultiRocket | 2020-2021 | 随机卷积+池化 | 公开 ✅ | scout E06 已列 (时域 motif, 非频域) |
| catch22 | 2019 | 22 个时序摘要 | 公开 ✅ | scout E06 已列 |

### F. OOD / 漂移
| 论文 | 年份 | 核心思想 | 源码 | 与本赛关联 |
|---|---|---|---|---|
| RevIN | 2022 (ICLR) | 逐样本反实例归一化 | 公开 | scout 已列; 注意 E01 ReVol-lite 归一化路线 PSEUDO +0.0011 gate 未过 → 归一化单独不够 |
| TabReD | 2024 | 时间切分下排名翻转; 简单模型仍强 | — | E03 稳定性门禁依据 (scout 已列) |
| Test-Time Training | 2020 | 测试时自监督微调 | — | 备选 (未入 Top3: 伪门禁过拟合风险) |

## 7. Final Verdict

| 候选 | 期望增益 | 置信 | 依据 |
|---|---|---|---|
| **COC** (余弦最优校准) | **+0.001 最可能** (区间 +0.0005~+0.003) | 中-高 | 直接实现度量最优解 (定理 1-5); 全部证据来自本地实测 (D1-D3、0.43 vs 0.156 幅度差、74.4% 集中度); 30min 出结果; E01 先例 (同类校准/表示增益 ~+0.001 量级) 给出现实上界 |
| **ETCI** (事件时条件创新) | **+0.003 最可能** (区间 0~+0.003) | 中-低 | 唯一还有未测信息面 (事件时间) 的方向; When Does Order Flow Matter (2026) + lo/hi 分层双证据; 但 P4-06A (0.0000) 与 E01 (+0.0011) 先例压住上限, B-lite v2 不通过则整线归零 |
| **SPCE** (谱相位相干编码) | **小概率 +0.01** (区间 0~+0.01) | 低 | 唯一直接攻击相位结构 (surrogate 最致命一刀, 96% 损失) 且带抗漂移理论 (形变稳定) 的表示; 若"跨通道同步+相位形态"实锤为真, 谱表示与一切现有特征正交 → blend 杠杆最大; 但 30s 窗谱估计方差大, 先验低 |

**排序依据 (诚实的概率账)**: COC 的成功概率最高但幅度受 E01 先例限制 (~+0.001 量级, 尾部专家完整版可能 +0.003); ETCI 条件化方向有双证据但本地负面先例也多 (P4-06A/P3-02/P4-01a), 且依赖 B-lite v2 通过; SPCE 概率最低但若成立是唯一可能跨台阶的 (相位是 surrogate 中最致命的信息维度, 而它完全无特征化)。

**一句话**: 先花 30 分钟跑 COC probe (免费捡 +0.001), 再花半天跑 ETCI 三臂 (决定条件模型线生死), SPCE 用 2-4 小时并行验证 (长线彩票) — 三者互不阻塞, 共享 PSEUDO 门禁。

## 8. 已核验一手来源 (本次)

- HLOB: "HLOB – Information persistence and structure in limit order books", ESWA 2024, DOI 10.1016/j.eswa.2024.126078, arXiv 2405.18938 ✅ (OpenAlex + arXiv 双实测)
- Deep LOB Forecasting (HLOB 前作): arXiv 2403.09267, Briola/Bartolucci/Aste ✅
- TLOB 官方代码: github.com/LeonardoBerti00/TLOB ✅ (GitHub API 实测)
- Maximum-entropy Scattering Models for Financial Time Series: ICASSP 2019, DOI 10.1109/icassp.2019.8683734 ✅
- When Does Order Flow Matter? State-Dependent L2 Liquidity-State Transitions in Crypto Futures: arXiv 2607.09230, Jeon, 2026 ✅ (OpenAlex 实测全标题)
- LOB Benchmark Study: arXiv 2308.01915 ✅
- plan v1.8.0 已核验 8 篇 (Cont 2010 / TLOB / LOB Repr / LOBERT / Order Flow Matter / State-dependent Hawkes / TradeFM / Event-based LOB Sim) ✅
- 仓库 grep: 152 特征族无谱/相干/小波特征; 唯一 FFT = p5_02i 破坏实验 ✅
- **外部核验脚本 abs_verify.py 追加 (2026-08-14, arXiv abs 页直抓)**: 1011.6402 Cont-Kukanov-Stoikov ✅ | 1612.09328 Neural Hawkes (Mei-Eisner) ✅ | 1709.07871 FiLM (Perez) ✅ | 1609.09106 HyperNetworks (Ha-Dai-Le) ✅ | 1903.07291 SPADE (Park, 条件归一化图像先例) ✅ | 1704.04110 DeepAR ✅ | 2002.09291 Transformer Hawkes (Zuo) ✅ | 2202.03528 TACTiS ✅ | 1502.04592 Hawkes in Finance (Bacry) ✅ | 1808.03668 DeepLOB (DOI 10.1109/TSP.2019.2907260) ✅
- **该批 6 个 ID 错配无效 (实测为无关论文, 引用清单中应删除)**: 1906.05964, 2102.01286, 2002.06569, 2203.01424, 2403.12543, 1908.03475 — 已 grep 确认不出现在本报告及仓库 docs 中
- 未确认项 (诚实标注): HLOB 官方仓库链接 (仅社区复刻); "LOBCAST" 命名基准 (arXiv/OpenAlex 均未命中, 以 2308.01915 benchmark study 代替); G-Research 前排方案细节 (Kaggle 页为 JS shell, 以仓库侦察文档为准)
