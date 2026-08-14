这是GPT1的方案：
我按你这份任务书把方向重新筛了一遍。你当前的证据已经足够强，不应该再围绕普通 TCN、Transformer、target residual 或 `[152 features, market latent]` concat 继续消耗实验额度；这些方向不是缺调参，而是 **inductive bias / information decomposition 不对**。尤其是 market 的有效信号已经被定位到 **≤30 秒、跨通道同步、非线性 phase/shape、近似时间反演对称**，同时 market 对 (|y|) 的预测显著强于方向。 

# Executive conclusion

如果现在只有 **3 次真正的大实验机会**，我会严格选：

| 排名    | 研究代号         | 核心                                                       | 我认为的定位                            |
| ----- | ------------ | -------------------------------------------------------- | --------------------------------- |
| **1** | **SCOPE-CI** | State-Conditioned Order-flow/Trade Innovation            | **最可能形成下一次 +0.003 台阶**            |
| **2** | **MAGNET**   | Magnitude-Conditioned Direction Mixture                  | **成功概率最高，最容易先拿 +0.001**           |
| **3** | **REVCOH**   | Reversal-Invariant Cross-Channel Phase/Coherence Encoder | **风险最大，但最像真正 +0.01 breakthrough** |

这三条分别对应三个尚未被充分利用的东西：

[
\text{SCOPE-CI}: I(Y;O|M),\ I(Y;T|M,O)
]

[
\text{MAGNET}: \text{Market 对 magnitude 的强信息}
+\text{ magnitude-dependent directional alpha}
]

[
\text{REVCOH}: I(Y;M)\text{ 中尚未被普通 sequence model 正确编码的 phase/synchronization}
]

而不是继续换模型名字。

---

# Similar competitions / datasets

下面的“相似度”是我按 **数据生成机制** 主观打分，不是按比赛名称。

| 排名     | Dataset / Competition                            |                Market |                    Orders |             Trades | Horizon / Target                   | Metric               | Drift        |     相似度 |
| ------ | ------------------------------------------------ | --------------------: | ------------------------: | -----------------: | ---------------------------------- | -------------------- | ------------ | ------: |
| **1**  | **LOBSTER / Nasdaq TotalView-ITCH**              |                 ✓ 全深度 |    ✓ 原始 submission/cancel |        ✓ execution | 自定义 future return / LOB target     | 自定义                  | 可构造长期        | **9.8** |
| **2**  | **CME BTC Futures Level-3 MBO / ByteGen corpus** |          ✓ full depth |        ✓ individual order |       ✓ executions | 原始 event stream，可自行标未来收益           | generative benchmark | 有            | **9.4** |
| **3**  | **Optiver Realized Volatility**                  |                ✓ book |                         — |           ✓ trades | 前10min → 后10min volatility         | RMSPE                | 明确时序 test    | **9.0** |
| **4**  | **CME Level-2 + matched trades**                 |                 ✓ 前5档 |            △ book changes | ✓ exchange-matched | 可自定义                               | simulation           | 有            | **8.8** |
| **5**  | **Market-by-Order forecasting corpus**           |                 ✓ LOB | ✓ individual instructions |               ✓/隐含 | 高频 future price movement           | classification       | 多资产          | **8.6** |
| **6**  | **Binance BTC/ETH L2 + trade flow 2023–26**      |               ✓ top20 |                         — |                  ✓ | liquidity-state transition         | classification       | **强，跨年**     | **8.3** |
| **7**  | **Optiver Trading at the Close**                 | ✓ orderbook + auction |                         △ |                  △ | 60s future relative WAP move       | MAE                  | 时序           | **7.9** |
| **8**  | **FI-2010**                                      |        ✓ 10-level LOB |                         — |                  — | future mid-price direction         | F1/classification    | 10 days      | **7.3** |
| **9**  | **Coinbase Deep Orderbook**                      |                 ✓ LOB |                         — |                  △ | 2s future price direction          | accuracy             | walk-forward | **7.0** |
| **10** | **DSLOB**                                        |       ✓ synthetic LOB |                         △ |                  △ | forecasting under controlled shift | task-dependent       | **专门设计 OOD** | **6.9** |

最像 MSCapital 的其实不是 FI-2010，而是 **LOBSTER / MBO 类原始 message data**。LOBSTER 同时提供 order-book snapshots 和 submissions、cancellations、executions，并保留 timestamp、order id、price、size、side，结构上几乎就是你这里 `market + order + transaction` 的未压缩版本。([Lobster Data][1])

ByteGen 所用的 CME Bitcoin futures 数据甚至是 Level-3 Market-by-Order：完整深度和 individual order tracking，虽然论文目标是生成而非预测，但其**数据生成层级**与 MSCapital 的原始 order stream 极接近。([arXiv][2])

Optiver Realized Volatility 反而在另一个维度高度相似：它同时有 book 与 trade 数据，用前 10 分钟预测未来波动。你的 market 现在最强的能力恰恰是判断“未来会不会大波动”，所以这个比赛比一般 LOB direction benchmark 更值得迁移经验。([Kaggle][3])

2021 的 Market-by-Order 工作尤其重要：作者明确发现 individual order instruction 所含信息与 LOB snapshot 并不完全重复，MBO 与 LOB 模型 ensemble 能进一步提高预测。这给你寻找 (I(Y;O|M)) 提供了很直接的实证支持。([arXiv][4])

另外，2026 年 Binance BTC/ETH 的研究得到一个与你的 Conditional Innovation 假设非常接近的结论：**先建 market state，再看 order flow 是否在 state 之上产生增量**；order flow 的价值依赖于当前 liquidity state，而且不同资产/状态下并不稳定。([arXiv][5])

---

# Literature map

| 方向                               | 文献                                |      年份 | 真正值得拿走的东西                                      | 源码                  |
| -------------------------------- | --------------------------------- | ------: | ---------------------------------------------- | ------------------- |
| Market-conditioned order flow    | Queue-Reactive Hawkes             |    2019 | event intensity 同时依赖当前 book state + past flow  | 数学模型                |
| Granular information             | Deep Learning for Market by Order |    2021 | MBO 相对 LOB 是 additive information              | pipeline 思路         |
| State-conditioned events         | State-Dependent Neural Hawkes     |    2022 | state → event intensity / type                 | 有公开研究实现生态           |
| Conditional generation           | Conditional Generators for LOB    |    2023 | **给定 state 生成 order flow**                     | 有相关实现               |
| Robust LOB benchmark             | LOBFrame                          | 2024–25 | 跨股票、跨时期 generalization；pipeline 完整             | **官方 GitHub**       |
| Higher-order channel interaction | HLOB                              | 2024–25 | 显式建 volume level 高阶相互作用                        | **LOBFrame**        |
| Conditional order generation     | TRADES                            |    2025 | market-state-conditioned order-flow generation | **DeepMarket 官方代码** |
| Simple spatial/temporal mix      | MLPLOB / TLOB                     |    2025 | 简单 MLP mixing 已可很强；复杂序列模型不天然必要                 | **官方代码**            |
| Symmetric local filters          | PRISM                             | 2025/26 | multi-resolution **symmetric FIR**，低参数量        | **官方 GitHub**       |
| Phase–amplitude modeling         | APRNet / TimeAPN                  | 2025/26 | 显式处理 phase/amplitude 与 nonstationarity         | 论文                  |
| Wavelet representation           | Kymatio / phase harmonics         |      成熟 | 固定局部 wavelets、phase harmonic                   | **官方代码**            |
| OOD benchmark                    | DSLOB                             |    2022 | 专门测试 distribution shift                        | 数据/benchmark        |

Queue-Reactive Hawkes 最关键的思想不是 Hawkes 本身，而是：**order-arrival distribution 必须 conditional on current book state**。后来 2024 的扩展甚至把 event type、arrival intensity 和 size 都建成 current state 的函数。([arXiv][6])

TRADES 又进一步直接学习“**market state 条件下的 order-flow distribution**”，而且官方 DeepMarket 仓库已经公开。因此我不建议你复制它的大 Transformer/diffusion；我建议**偷它的问题定义，扔掉它的大模型**。([arXiv][7])

HLOB 则说明 LOB 信息的重要部分可以来源于价格/volume level 之间的高阶依赖，而不是每个 channel 各自的 temporal trend；它与目前“channel desynchronization 几乎摧毁 market alpha”的发现高度一致。([arXiv][8])

反直觉方向里，PRISM 用对称 FIR kernel 构造轻量多尺度时序表示，并有官方实现；APRNet 则显式建模 amplitude–phase interaction。这两类 inductive bias 都比“大 Transformer 看 200 步序列”更符合你已经定位出来的信号。([arXiv][9])

---

# 一个很重要的数学结论：Cosine 到底应该预测什么？

你特别要求严格回答这个。

假设 leaderboard 是对整个预测向量计算一次：

[
C(\mathbf p,\mathbf y)
======================

\frac{\mathbf p^\top\mathbf y}
{|\mathbf p||\mathbf y|}
]

给定整个 test feature matrix (X)，我们要最大化期望：

[
E[C\mid X]
==========

\frac{\mathbf p^\top}
{|\mathbf p|}
E\left[
\frac{\mathbf Y}{|\mathbf Y|}
\mid X
\right]
]

令

[
\mathbf q(X)=
E\left[
\frac{\mathbf Y}{|\mathbf Y|}
\mid X
\right]
]

由 Cauchy–Schwarz：

[
E[C|X]\leq |\mathbf q(X)|
]

等号成立当且仅当

[
\boxed{\mathbf p^*(X)=c\mathbf q(X),\quad c>0}
]

所以**严格有限样本 Bayes 解**确实是：

[
\boxed{
\mathbf p^*
\propto
E[\mathbf Y/|\mathbf Y|\mid X]
}
]

注意：这是**整个 target vector 先做 norm**，不是对单个 scalar 写：

[
E[y/|y|\mid x]
]

后者基本就是预测 sign，明显不是这里的答案。

但是 test sample 很多时，

[
|\mathbf Y|
\approx
\sqrt{\sum_i E[Y_i^2|x_i]}
]

norm 会集中成为近似公共因子，因此：

[
q_i(X)
\approx
\frac{E[Y_i|x_i]}
{\text{common scale}}
]

于是实际可实现的 per-sample 模型的 Bayes 解变成：

[
\boxed{p(x)\propto E[Y|X=x]}
]

也就是说：

> **Cosine 并没有推翻 conditional mean。**

真正值得改变的是：**怎么估计 (E[Y|x])**。

这正好引向你的 magnitude/sign 发现。

令

[
Y=S\cdot A,\qquad
A=|Y|,\quad S\in{-1,+1}
]

那么精确有：

[
\boxed{
E[Y|x]
======

\int
a
\left[
2P(S=+1|A=a,x)-1
\right]
p(a|x),da
}
]

只有在

[
S\perp A\mid X
]

时才能粗略写成：

[
E[A|x]\times E[S|x]
]

但你的实验已经显示：小 (|y|) 时方向准确率约 0.37，而最大的 (|y|) 样本达到约 0.585。也就是说 **direction 与 magnitude 显然存在条件依赖**。

所以正确架构不是：

```text
magnitude × one universal direction model
```

而是：

```text
magnitude distribution
        ↓
condition / gate
        ↓
different directional alpha
        ↓
integrate over magnitude
        ↓
E[y|x]
```

这就是下面的 MAGNET。

---

# Top 1 — SCOPE-CI

## State-Conditioned Orthogonalized Process Events

### 核心思想

不是预测：

[
y-\hat y_{baseline}
]

所以它**不是你已经失败 5 次的 residual modeling**。

而是：

[
O^\perp=O-E[O|M]
]

再：

[
T^\perp
=======

T-E[T|M,O]
]

真正问：

> **“在这种 market state 下，这 60 秒 order flow 有多不正常？”**

以及：

> **“看到 market + orders 后，这些 trades 又有多意外？”**

这在微观结构上有非常扎实的对应：queue-reactive 模型明确让 event rate conditional on book state；TRADES 直接生成 conditional-on-market-state order flow；经典 event-impact 工作也把冲击拆成即时作用、对未来事件率的改变等 history-dependent 部分。([arXiv][6])

### Architecture

完整版我建议：

```text
Market: last 10~20 steps
        │
relative/sample normalization
        ↓
Local State Encoder
        ↓
S ∈ R^64
        │
        ├── Conditional Order Model
        │     μO(S), σO(S)
        │
Observed Order features
        ↓
zO = (O - μO) / σO
        ↓
FiLM-conditioned Order Encoder
        ↓
hO
        │
        ├── Conditional Trade Model
        │     μT(S,hO), σT(S,hO)
        │
Observed Trade
        ↓
zT = (T - μT) / σT
        ↓
Trade Innovation Encoder
        ↓
hT

[S → gated hO → gated hT]
        ↓
low-capacity RealMLP-style head
        ↓
ŷ
```

重点不是网络，而是 **两次 conditional subtraction / standardization**。

最好让 nuisance model 输出：

[
(\mu,\log\sigma)
]

于是 surprise 不只是：

[
O-\hat O
]

而是：

[
z_O=
\frac{O-\mu_O(M)}
{\sigma_O(M)+\epsilon}
]

这会比普通 residual 更抗 regime scale drift。

### 新信息

主要攻击：

[
\boxed{I(Y;O|M)}
]

和：

[
\boxed{I(Y;T|M,O)}
]

这是三条路线里最明确能回答：

> “到底引入了哪个以前没有使用好的随机变量？”

的一条。

### 为什么不是 concat 换皮

旧：

```text
market latent
order features
trade features
→ concat
```

要求最终 MLP 自己学会“什么 order flow 在什么 market state 下才异常”。

SCOPE：

```text
market
→ expected order distribution
→ observed - expected
```

**conditional comparison 在进入 alpha model 之前就已经完成。**

这是信息结构变化，不是 fusion layer 变化。

### OOD

这是它最大的优点，也是最大的风险。

优点：

```text
absolute order flow
↓
relative surprise under current state
```

比直接记忆 month 0–70 中“大成交量是多少”更稳定。

这与 DAIN/stationary LOB feature 这类“相对表示优于 absolute regime”的研究方向一致。([arXiv][10])

风险是：

[
P(O|M)
]

本身也可能从 train 漂到 test。

所以必须：

* sample-wise relative normalization；
* cross-fitting nuisance model；
* 控制 nuisance capacity；
* 同时输出 conditional scale；
* monthly OOS 测 surprise distribution 是否稳定。

你的 train/test 跨越 month 0–70 到 71–108，之前 TCN 已出现 val gain +0.0044、LB 直接跌到 0.082，因此任何 sequence gain 如果没有 temporal stability 都不能信。

### 2–6h probe

**不要上 raw event Transformer。**

直接从现有 152 features 中拆出：

```text
20–40 Order dynamics
10–30 Trade dynamics
```

训练三个小模型：

[
\hat O=g(M)
]

[
\hat T=h(M,O)
]

生成 **OOF**

[
z_O,\ z_T
]

然后：

```text
existing v7/RealMLP features
+
zO
+
zT
→ same RealMLP
```

这里 probe 阶段允许 concat，因为目的只是回答：

> innovation 有没有 alpha？

不是最终 fusion 方案。

### Kill criterion

我会直接写死：

**继续：**

[
\Delta cosine\ge +0.0010
]

且至少 3 个后段 temporal fold 中 2 个为正；

并且 new prediction 与 baseline：

[
corr<0.95
]

最好 <0.90。

**KILL：**

* Δcos < +0.0005；
* 增益只出现在随机 CV；
* 后 20% 时间段为负；
* `corr(new,baseline)>0.97`；
* innovation feature importance 最终又只还原成原始 O/T magnitude。

---

# Top 2 — MAGNET

## Magnitude-Conditioned Alpha Network

这是我认为**应该最先跑**的实验。

因为你已经拥有异常强的先验证据：

* Market (|y|) corr ≈ 0.43–0.47；
* extreme P90 AUC ≈ 0.78；
* v7 的 (|pred|) 与 (|y|) corr 只有 ≈0.156；
* v7 在真正大波动样本上反而方向最好；
* top 20% 样本贡献约 74.4% cosine inner product，extreme 5% 贡献约 56.5%。 

这个组合实在太强，不利用非常可惜。

Quantile LOB 工作早就证明，直接学习未来 return distribution/quantiles 可以比单一点估计获得更稳定的信息。([arXiv][11])

### Architecture

不要：

```text
Market magnitude
×
v7
```

完整版应该：

```text
Market M
 ↓
Magnitude distribution head
 ↓
π1...πK
E[A | bin k, M]

                 Order + Trade
                      ↓
             shared alpha encoder
                      ↓
          direction expert d1
          direction expert d2
                  ...
          direction expert dK

πk(M) ───────────────┐
A_k(M) ──────────────┼→ Σ πk · A_k · dk
direction dk ────────┘
                     ↓
                    ŷ
```

例如 K=4：

```text
quiet
normal
large
extreme
```

最终近似的正是：

[
E[Y|x]
\approx
\sum_k
P(A\in B_k|M)
E[A|B_k,M]
E[S|A\in B_k,M,O,T]
]

而不是假设 magnitude 和 direction 独立。

### 新信息

它主要不是创造新的随机变量。

而是让当前被浪费的：

[
\boxed{I(Y;M)}
]

中的 magnitude 信息真正进入最终 decision rule。

同时利用：

[
P(S|A,x)\neq P(S|x)
]

这个你新发现的 conditional structure。

### OOD

最大风险是：

> train 的“大波动”与 test 的“大波动”尺度变了。

所以 gate **不要使用 absolute |y| threshold 的逻辑**。

最终应倾向：

* quantile/rank magnitude；
* sample-relative market input；
* gate probability clipping；
* low-capacity monotone calibration；
* 不让 regime ID、month ID 进入 gate。

### 最小 probe

甚至不需要重新训练大型模型。

已有：

```text
market magnitude/extreme OOF prediction m
v7 OOF prediction p
```

先拟合一个只有 3–5 个参数的：

[
\hat y=p\cdot g(m)
]

例如按 market extreme score 分成 4 个 OOF quantile，分别学习：

[
w_1,w_2,w_3,w_4
]

得到：

[
\hat y_i=w_{bin(m_i)}p_i
]

注意：**这只是验证“amplitude allocation 是否有效”的 probe，不是最终架构。**

### Kill criterion

这条我要求更严格，因为 evidence 已经很强：

**继续：**

[
\boxed{\Delta cosine\ge +0.001}
]

而且最后 1/3 temporal period 也必须 >0。

如果一个四参数 gate 连 +0.001 都做不到，那就说明：

> “Market 会预测 magnitude + v7 大波动方向更准”

虽然统计上成立，但无法转成 leaderboard gain。

这种情况下 **整个 MAGNET 路线直接降级**。

---

# Top 3 — REVCOH

## Reversal-Invariant Cross-Channel Coherence Encoder

这是反直觉方案，也是我最感兴趣的一条。

你的 destruction experiment 基本描述了一个信号处理问题：

```text
短窗口
+
多传感器/多通道
+
同步结构
+
相位关系
+
正反播放近似不变
```

它其实更像：

> **speech / vibration / multichannel signal recognition**

而不像普通金融时间序列 forecasting。

PRISM 的 symmetric multi-resolution FIR、APRNet 的 phase-amplitude interaction，以及 HLOB 的 higher-order cross-level dependencies，都为这种思路提供了对应 inductive bias。([GitHub][12])

### 一个重要修正

**我不推荐直接上普通 scattering transform。**

Kymatio 的经典 scattering 使用固定 wavelet + modulus，优点是局部稳定、低容量，但 modulus 会主动消掉相当一部分 phase 信息。([GitHub][13])

而你的 phase destruction 几乎杀掉所有 signal。

所以真正值得做的是：

# Cross-wavelet / phase coherence

而不是 plain scattering。

甚至 Kymatio 生态里已经有 **wavelet phase harmonics** 的实现，可作为原型参考。([GitHub][14])

### Architecture

只取：

[
T=10\sim12
]

也就是约 30–36 秒。

```text
Market[T,C]
   ↓
sample-wise relative normalization
   ↓
low-rank channel projection C → 16/24
   ↓
Symmetric FIR filter banks
kernel = 3 / 5 / 9
k[t] = k[-t]
   ↓
complex/quadrature local coefficients
Z[f,t,c]
   ↓
Cross-channel coherence / Gram
Z Z*
   ↓
Re(cross-spectrum)
|Im(cross-spectrum)|
phase-lock strength
local covariance
   ↓
pool over t
   ↓
small MLP
   ↓
market alpha
```

为什么这么设计？

对于两个通道：

[
Z_i(f,t),Z_j(f,t)
]

看：

[
Z_i Z_j^*
]

它同时携带：

* amplitude co-movement；
* relative phase。

取：

[
\Re(Z_iZ_j^*)
]

以及：

[
|\Im(Z_iZ_j^*)|
]

可以保留很多**相对 phase synchronization**，同时又比 absolute phase 更接近 reversal invariant。

再显式做：

[
h_{inv}(x)
==========

\frac{
h(x)+h(reverse(x))
}{2}
]

这与 destruction experiment 不是“碰巧契合”，而是把它直接写进 architecture。

PRISM 的对称 FIR 已证明可以用很少参数建立 multi-resolution representation，而且官方实现可直接借结构。([arXiv][9])

### 为什么不是 TCN 换皮

TCN：

```text
learned temporal convolution
→ stack
→ directional temporal representation
```

REVCOH：

```text
symmetric local filter
→ complex/local shape
→ channel × channel bilinear interaction
→ phase coherence
→ reversal-invariant pooling
```

真正的核心算子已经不是 temporal convolution，而是：

[
\boxed{\text{cross-channel local second-order / phase interaction}}
]

TCN failed 不足以否定它。

### 新信息

仍然属于：

[
I(Y;M)
]

但问题变成：

> 当前 market encoder 是否根本没把那个随机变量表示出来？

你的现有 market cosine corr 只有约 0.086，但 destruction 表明原始 sequence 中存在极强、结构很具体的信息。

这就给了换 inductive bias 的理由。

### OOD

三条路线里它理论上最抗 drift：

* local ≤30s；
* relative normalization；
* symmetric filters；
* 不用绝对 regime；
* 不记忆 200-step long signature；
* low parameter count；
* coherence 比 absolute price/volume 更 scale invariant。

这正好对应你要求优先考虑的 relative/local/invariant representation。

但风险也最大：

> “phase destruction 很重要”不等于“一定是 Fourier phase”。

它也可能只是随机 phase 破坏顺便毁掉了局部 cross-channel shape。

所以必须先 probe。

### 2–6h probe

完全不用神经网络。

取最近 10 timestep，对选出的 channel group 算：

1. local covariance / correlation；
2. pairwise cross-correlation lag (-3...+3)；
3. 做 reversal symmetric：

[
r_{ij}^{sym}(k)
===============

r_{ij}(k)+r_{ij}(-k)
]

4. 2–3 个短频率的 Fourier coefficient；
5. cross-channel：

[
\Re(Z_iZ_j^*),
\quad
|\Im(Z_iZ_j^*)|
]

6. 喂给现有 RealMLP。

这才是真正便宜的生死实验。

### Kill criterion

**继续：**

market-only：

[
corr(\hat y,y)>0.09
]

或者虽只有相近 0.086，但与现有 market model：

[
corr<0.7
]

且 blend：

[
\Delta cosine>+0.0007
]

同时后期 temporal folds 不退化。

**KILL：**

* corr <0.06；
* 与现有 market pred corr >0.9；
* phase/coherence features 被 covariance features 完全替代；
* reverse augmentation 完全没有任何稳定作用。

---

# 最值得先跑的实验

虽然 SCOPE 是我的**方案排名 #1**，实验执行顺序我会反过来按照“信息价值 / 计算成本”排序。

## Experiment 1 — MAGNET Gate Probe

**Hypothesis**

Market 的 magnitude alpha 可以重新分配 v7 prediction norm，让 cosine 更集中到真正值得下注的样本。

**Implementation**

```text
OOF market extreme/magnitude score
+
OOF v7
→ 4-bin learned scaling
```

只有约 4 个参数。

**Baseline**

v7 / 当前 0.142 ensemble。

**Metric**

* global cosine；
* monthly cosine；
* largest-|y| decile cosine contribution；
* correlation with baseline。

**Expected**

[
+0.001\sim+0.002
]

**Kill**

[
\Delta<+0.001
]

或者 temporal 后段不稳定。

**Next**

成功才上 full magnitude-conditioned direction experts。

---

## Experiment 2 — SCOPE Innovation Probe

**Hypothesis**

现有 152 features 混合了 predictable flow 与 true unexpected flow；conditional residualization 能释放：

[
I(Y;O|M), I(Y;T|M,O)
]

**Implementation**

```text
M → predict selected O features
O innovation

M + O → predict selected T features
T innovation

innovation → existing RealMLP
```

必须 OOF / cross-fit。

**Baseline**

同 feature budget 的 raw O/T RealMLP。

**Metric**

* Δ cosine；
* monthly consistency；
* innovation vs raw feature importance；
* new prediction correlation；
* late-period CV。

**Expected**

如果成立：

[
+0.0015\sim+0.004
]

**Kill**

Δ < +0.0005，或 late folds 为负。

**Next**

成功后再做 FiLM/conditional density/full hierarchical architecture。

---

## Experiment 3 — REVCOH Deterministic Probe

**Hypothesis**

Market alpha 本质上是局部 multichannel phase/coherence pattern，而不是 conventional temporal dynamics。

**Implementation**

```text
last 10 market timesteps
→ symmetric lag correlation
→ local covariance
→ cross-spectrum/coherence
→ RealMLP
```

**Baseline**

current market-only cosine model。

**Metric**

* corr with y；
* corr with existing market prediction；
* blend Δ；
* reversed-input stability；
* monthly consistency。

**Expected**

正常成功：

[
+0.0005\sim+0.002
]

但真正值得兴奋的结果不是单模型分数，而是：

```text
market corr >= current model
AND
pred corr < 0.7
```

这意味着你真的发现了一个新 market representation。

**Kill**

没有 orthogonality 或 temporal OOD 明显。

**Next**

才实现 symmetric FIR + learned cross-channel coherence network。

---

# Final verdict

### 最可能 +0.001

[
\boxed{\text{MAGNET}}
]

因为几乎所有 prerequisite 都已经在你的 diagnostic 中被验证了，而且 probe 只需要几个参数。它是现在**风险收益比最高**的一步。

### 最可能 +0.003

[
\boxed{\text{SCOPE-CI}}
]

这是我认为真正有资格叫“新信息源”的方向。MBO 研究已经说明 raw order instructions 可以提供 LOB 之外的 additive alpha，而 conditional LOB literature 又告诉我们 order behavior 本来就应该在 market state 条件下解释。([arXiv][4])

### 小概率真正 +0.01 breakthrough

[
\boxed{\text{REVCOH}}
]

不是因为 wavelet 很新，而是因为你自己的 destruction experiment 已经给出了一个非常罕见、非常具体的 fingerprint：

[
\boxed{
\text{short-range}
+
\text{cross-channel synchronous}
+
\text{phase-sensitive}
+
\text{time-reversal symmetric}
}
]

这套 fingerprint **天然不像 Transformer 问题**，反而非常像多通道信号识别 / coherence detection。

如果 REVCOH 的廉价 deterministic probe 就能达到甚至超过当前 market sequence 模型，同时两者相关性很低，我会把它立即升成全项目最高优先级。

而 **Conditional Innovation 假设本身我不会否定**。经过这轮文献调查，它实际上比最初看起来更有理论支持；但我会把它具体化为：

[
\boxed{
\text{conditional event surprise}
\neq
\text{target residual}
}
]

这是最重要的区别。

你现在最不应该做的，是再找一个“2026 最新 LOB Transformer”训练两天。TLOB 自己的结果都在提醒这个领域：简单 MLP 适配好输入结构后就能非常强，而 LOBFrame 的跨时期研究同样强调 microstructure generalization 与 conventional validation performance 并不是一回事。([arXiv][15])

**所以我会先跑 Experiment 1 的 4-bin MAGNET probe；它是三条路线里最快能告诉我们“0.142 后面有没有现成的 0.143–0.144”的实验。**

[1]: php.lobsterdata.com/info/WhatIsLOBSTER.php?utm_source=chatgpt.com "what is LOBSTER. - LOBSTER"
[2]: arxiv.org/html/2508.02247v1?utm_source=chatgpt.com "A Tokenizer-Free Generative Model for Orderbook Events ..."
[3]: kaggle.com/code/gunesevitan/optiver-realized-volatility-prediction-eda?utm_source=chatgpt.com "Optiver Realized Volatility Prediction - EDA"
[4]: arxiv.org/abs/2102.08811?utm_source=chatgpt.com "Deep Learning for Market by Order Data"
[5]: arxiv.org/abs/2607.09230?utm_source=chatgpt.com "When Does Order Flow Matter? State-Dependent L2 Liquidity-State Transitions in Crypto Futures"
[6]: arxiv.org/abs/1901.08938?utm_source=chatgpt.com "Queue-reactive Hawkes models for the order flow"
[7]: arxiv.org/html/2502.07071v1?utm_source=chatgpt.com "Generating Realistic Market Simulations with Diffusion ..."
[8]: arxiv.org/abs/2405.18938?utm_source=chatgpt.com "HLOB -- Information Persistence and Structure in Limit Order Books"
[9]: arxiv.org/abs/2508.04503?utm_source=chatgpt.com "PRISM: Lightweight Multivariate Time-Series Classification through Symmetric Multi-Resolution Convolutional Layers"
[10]: arxiv.org/abs/1902.07892?utm_source=chatgpt.com "Deep Adaptive Input Normalization for Time Series ..."
[11]: arxiv.org/abs/1906.04404?utm_source=chatgpt.com "Extending Deep Learning Models for Limit Order Books to Quantile Regression"
[12]: github.com/fedezuc/PRISM?utm_source=chatgpt.com "[Neurocomputing] PRISM: Lightweight Multivariate Time- ..."
[13]: github.com/kymatio/kymatio?utm_source=chatgpt.com "Kymatio: Wavelet scattering in Python"
[14]: github.com/kymatio/phaseharmonics?utm_source=chatgpt.com "PhaseHarmonics: Wavelet phase harmonic transform ..."
[15]: arxiv.org/abs/2502.15757?utm_source=chatgpt.com "TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data"


























这是Grok1的方案：**Executive conclusion**

If you only have 3 large experimental slots, run these in order:

1. **MagGate-Innov (Magnitude-Gated Conditional Innovation + cosine-optimal head)** — highest expected incremental value and best match to the empirical structure (Market magnitude >> direction; cosine dominated by extremes; v7 stronger on direction of large |y|; need I(Y; O|M) and I(Y; T|M,O)).
2. **SyncScat / Local Cross-Channel Shape Encoder** (scattering + local covariance / multi-scale shapelets on ≤10–12 timesteps, with explicit time-reversal invariance) — the strongest “new inductive bias” candidate that directly targets the surrogate destruction results.
3. **FiLM-State Conditioned Short-Horizon Event Residual** (Market state → FiLM/conditional normalization of short Order/Trade windows → innovation features → blend) — cleanest realization of the conditional information chain with good OOD properties via relative/conditional representations.

Do **not** start with another Transformer/TCN/long-sequence or plain residual-on-y experiment.

---

# Similar competitions / datasets (Top 10 by structural similarity)

| Rank | Dataset / Competition | Market (context) | Orders | Trades | Horizon | Target | Metric | Drift risk | Similarity notes |
|------|-----------------------|------------------|--------|--------|---------|--------|--------|------------|------------------|
| 1 | Optiver Realized Volatility Prediction (Kaggle 2021) | LOB snapshots (multi-level) + book features over ~10 min | Implicit via book | Explicit trade data | ~10 min realized vol | Volatility (magnitude) | RMSPE | High (time-id) | Closest: multi-channel book + trades → magnitude; strong feature-agg + NN wins |
| 2 | Optiver - Trading at the Close (Kaggle 2023–24) | Order book + auction book | Yes (imbalance, sizes, prices) | Implicit | 60s relative move | Price movement (near zero-mean) | MAE | High | Book + auction flow → short-horizon return; heavy feature engineering + NN blend |
| 3 | FI-2010 (benchmark) | 10-level LOB snapshots | Snapshot only | No raw events | 10–100 ticks | Mid-price direction (3-class) | Accuracy / F1 | Medium (short period) | Canonical LOB spatial-temporal; DeepLOB / TLOB / HLOB / LiT baselines |
| 4 | LOBSTER-derived (NASDAQ, academic) | Full event LOB + reconstructions | Full add/cancel | Full executions | Seconds–minutes | Mid-price / impact / direction | Various | High | Closest event-stream structure to your O + T |
| 5 | Jane Street Market Prediction / Real-Time Market Data Forecasting | Anonymized features (market state) | No raw | No raw | Multi-horizon responders | Returns / responders | Utility / correlation-like | Very high | Feature-rich, strong OOD, ensemble culture |
| 6 | G-Research Crypto Forecasting | Crypto LOB / trades (some notebooks) | Partial | Yes | Minutes–hours | Returns | Weighted correlation | Extreme | High-frequency crypto microstructure |
| 7 | High-frequency crypto LOB (Bybit / public Kaggle sets, FinRL etc.) | Multi-level LOB | Snapshots / events | Yes | Seconds–minutes | Direction / volatility | Accuracy / PnL | Extreme | Good for short-horizon cross-channel tests |
| 8 | Ubiquant Market Prediction | Tabular market features | No | No | Daily-ish | Returns | Pearson | High | Correlation metric culture, ensemble |
| 9 | Two Sigma / Winton / older quant challenges | Mixed | Rare | Rare | Varied | Returns / vol | Varied | High | Historical reference for feature + blending |
| 10 | Hull Tactical / TradeMaster / M6-style | Macro + price | No | No | Daily+ | Allocation / relative | Sharpe / WMAE | Medium | Less microstructure, more regime |

**Key structural match criteria used**: presence of multi-channel market state + order/trade events, short-horizon target related to price change or magnitude, evaluation that cares about extremes or correlation/cosine-like quantities, and temporal drift.

Optiver Realized Volatility is the single most useful external reference for magnitude modeling + book features. Optiver Close and FI-2010/LOBSTER are the best for book/event architecture ideas.

---

# Literature map (selected, 2022–2026 focus + classics)

**LOB / microstructure**
- DeepLOB (2018/2019): CNN spatial + Inception + LSTM. GitHub: zcakhaa/DeepLOB. Input ≈ (T, 40) or similar for 10-level. Strong baseline but your TCN/Transformer negatives already warn against naive long sequence.
- TLOB (2025, arXiv:2502.15757): Dual spatial-temporal attention on patches. Code available (LeonardoBerti00/TLOB). Better on longer horizons / volatile regimes; still sequence-heavy.
- LiT (2025): Patch + Transformer, no heavy CNN. Spatial-temporal.
- HLOB (2024): Homological / information-filtering network on volume levels + CNN/LSTM elements. Emphasizes higher-order spatial structure among levels.
- TransLOB / earlier Transformers for LOB: causal conv + masked attention.

**Event / point process**
- Neural Hawkes / marked point processes: generally long-range; low priority given your reverse-sequence and block-shuffle results.

**Volatility / magnitude + direction**
- Heteroscedastic / distributional regression, mixture density, quantile, extreme-event models. Optiver Vol solutions heavily used realized-vol style aggregations + NN.

**Conditional / multimodal**
- FiLM (Feature-wise Linear Modulation) applied to sequential encoders + tabular/market conditioning (recent finance applications show IC gains). Cleaner than concat.
- Conditional normalizing flows / conditional density estimation for innovation.

**Signal-processing / shape (anti-intuitive)**
- Scattering transform (Mallat lineage): wavelet cascade + modulus → stable multi-scale phase/energy coefficients; time-shift and deformation stable; used on financial series for multifractal / heavy-tail capture. Maximum-entropy scattering models exist for financial TS.
- Shapelet learning / ShapeFormer / recent financial shapelet frameworks (2024–2025): discriminative local shapes, scale/shift robust, interpretable; multivariate extensions.
- Local covariance / Gram matrices, cross-channel coherence, phase synchronization measures (neuroscience / dynamical systems transfer).

**Cosine / correlation metrics**
- Uncentered cosine is scale-invariant in the prediction. Optimal point prediction under expected cosine is the direction of \(\mathbb{E}[y/\|y\| \mid x]\) (unit-vector expectation). Scale of output can be fixed (e.g., 1) or used only for secondary weighting. This is distinct from \(\mathbb{E}[y \mid x]\).

---

# Top 3 candidate architectures

### 1. MagGate-Innov (Priority 1)
**Codename**: MagGate-Innov  
**Sources**: Heteroscedastic / distributional regression + Optiver-vol style magnitude modeling + FiLM-style conditioning + information-theoretic residualization + cosine decision theory. No single paper is a full copy; this is a re-design around your diagnostics.

**Why it matches your facts**
- Market alone: strong \(\operatorname{corr}(\hat{m}, |y|) \approx 0.43–0.47\), weak direction.
- Cosine massively weighted toward top |y| (top 5% ≈ 56% contribution).
- v7 already better directionally on large |y|.
- Explicitly targets \(I(Y;M)\) (magnitude/extreme) + \(I(Y;O|M)\) + \(I(Y;T|M,O)\) (directional innovation).
- Avoids residual-on-y (which failed) by residualizing the **event streams**, not the target.

**Architecture (implementable)**
- Market encoder (short): last ≤12 timesteps (≈30–36 s) of multi-channel market → local mixer (depthwise temporal conv or small MLP-Mixer / Inception) → state \(S\) + magnitude head \(m = P(|y| > \tau)\) or \(\mathbb{E}[|y| \mid M]\) or quantile.
- Order/Trade short windows (last 60 s, but emphasis on last 10–20 s): event features or binned sequences.
- Conditioned innovation: \(S\) produces FiLM (\(\gamma, \beta\)) or conditional LayerNorm parameters that modulate the Order and Trade encoders → expected behavior under market state → residual / surprise features (observed − predicted, or attention residual, or normalized innovation).
- Direction expert(s): innovation features (+ optional light v7 features) → direction logits or signed alpha \(d\).
- Cosine-optimal head: output \(p \propto d \cdot g(m)\) where \(g\) is a learned positive gate (soft, rank-based, or mixture). Or directly predict toward \(\mathbb{E}[y/\|y\| \mid x]\) (train with cosine or unit-vector loss). Scale of \(p\) can be fixed.
- Loss: primary uncentered cosine (or cosine on unit targets) + auxiliary magnitude loss (BCE on extremes / MSE on |y| / quantile) + optional orthogonalization regularizer between magnitude and direction heads.
- Normalization: sample-wise / relative (already known to help) + conditional norm.

**Why not covered before**: Not concat, not plain residual-on-y, not long-sequence model. Explicitly separates magnitude channel (Market) from conditional directional innovation and optimizes the exact decision rule for the metric.

**Expected new information**: \(I(Y;M)\) (magnitude/extreme probability) + pure innovations \(I(Y;O|M)\), \(I(Y;T|M,O)\).

**OOD risk**: Medium-low if everything is relative/conditional and capacity on absolute levels is kept low. Magnitude patterns may be more stable than absolute price levels.

**Minimal probe (2–6 h)**
- Train Market-only magnitude model (last 10–12 steps, simple local mixer or even RealMLP on hand-crafted short-window sync features).
- Compute corr with |y|, AUC on |y|>P90, and cosine contribution of top quantiles.
- Simple gate: multiply existing v7 by soft magnitude score; measure Δ cosine and monthly consistency.
- Kill if: magnitude corr < 0.35 on held-out months, or gated blend gives < +0.001 stable cosine, or strong month-to-month sign flip.

**Full version if probe lives**: full FiLM innovation + mixture-of-experts on regimes defined by \(S\) + cosine-optimal training.

### 2. SyncScat / Local Cross-Channel Shape Encoder (Priority 2 — anti-intuitive)
**Codename**: SyncScat  
**Sources**: Scattering transform (Mallat et al., financial scattering models), multivariate shapelets / ShapeFormer-style (2024–2025), local covariance / Gram matrices, phase-coherence ideas from dynamical systems / neuroscience.

**Why it matches**
- Surrogate results: time structure critical but **reversible**; effective scale ≤10 timesteps; channel desynchronization destroys almost everything; phase/shape destruction kills signal. Classic sequential inductive biases (causal LSTM/TCN/Transformer) are poorly matched; scattering and local shape/covariance are natural for short-range, multi-channel, phase-sensitive, deformation-stable patterns.

**Architecture**
- Input: Market tensor last ≤12–16 timesteps × channels (bid/ask/depth/spread/volume etc.), sample-wise normalized.
- Scattering or multi-scale wavelet filter bank (order 1–2) per channel + cross-channel products or joint scattering; or learnable shapelet bank (multi-length, multi-channel) with DTW-tolerant or correlation distance.
- Explicit time-reversal invariant pooling: \(f(x) + f(\mathrm{reverse}(x))\) or even/odd decomposition + invariant readout.
- Local covariance / Gram features across channels inside sliding short windows as additional channels.
- Lightweight readout (RealMLP-style or small MLP) → prediction. Can be used as pure Market model or as additional features/conditioned input to MagGate.
- Optional: turn the same encoder on short Order/Trade density matrices.

**Why not covered**: Path signatures and 2.5D grids were tried at insufficient scale/magnitude; this is a different inductive bias (multi-scale modulus/phase stability + explicit invariance) aimed exactly at the surrogate findings. Not a Transformer/TCN.

**Expected new information**: Higher-quality extraction of \(I(Y;M)\) via the precise geometric object the surrogates identified (short-range cross-channel nonlinear phase patterns).

**OOD risk**: Lower than sequential models — scattering coefficients and relative shapes are more invariant to absolute regime shifts.

**Minimal probe**
- Fixed (non-learned) scattering or simple multi-scale local covariance + shapelet distances on last 10–12 steps → RealMLP or linear.
- Compare cosine / corr vs current Market baseline and vs shuffled-channel control.
- Kill if: improvement over current Market < +0.01 cosine on validation **and** no clear gain on extreme |y| subset; or coefficients highly correlated (>0.9) with existing 152 features.

### 3. FiLM-State Conditioned Short-Horizon Event Residual
**Codename**: CondInnov-FiLM  
**Sources**: FiLM conditioning literature + conditional density / residualization ideas + short-horizon LOB event modeling.

**Architecture sketch**
- Market → compact state \(S\) (same short encoder as above).
- \(S\) → FiLM generator (\(\gamma,\beta\)) applied inside Order and Trade encoders (or conditional Batch/LayerNorm).
- Predict “expected” short-horizon order-flow / trade-response statistics under \(S\); form innovation = observed – expected (or normalized residual / attention residual).
- Innovation features + \(S\) → head (or feed into MagGate direction expert).
- Keep windows short; prefer relative/event-rate features.

**Why it matches the information chain** and is not plain concat or residual-on-y.

**OOD**: Conditional normalization + relative innovations should be more robust than absolute sequence models.

**Probe**: Market state → predict a few key Order/Trade summary statistics; measure residual predictability of y after controlling for Market; kill if residual cosine gain < 0.002–0.003 or residuals collinear with existing features.

---

# Most worth running first (ordered)

**Experiment 1 – Magnitude diagnostic + simple gate (fastest kill/live)**  
- Hypothesis: Market magnitude signal is the highest-leverage unused (or under-used) information for the cosine metric.  
- Implementation: short-window Market → magnitude / extreme probability; soft-gate existing v7 (or produce \(p = d_{\mathrm{v7}} \cdot g(m)\)).  
- Baseline: current 0.142 blend.  
- Metrics: overall cosine, cosine on |y| quantiles, monthly stability, corr(new, baseline).  
- Expected: +0.001–0.003 if gate is well-calibrated.  
- Kill: < +0.001 stable or degrades extremes.  
- Next: if live, replace simple gate by learned MagGate-Innov.

**Experiment 2 – SyncScat probe**  
- Hypothesis: explicit short-range cross-channel phase/shape features extract cleaner Market alpha than previous sequence or feature attempts.  
- Implementation: fixed scattering or local Gram + multi-scale shape distances on ≤12 steps → small model.  
- Kill criteria: no material lift on Market-only cosine or on extreme subset; high redundancy with 152 features.  
- Next: if live, feed as additional conditioning or features into MagGate.

**Experiment 3 – Conditional innovation residual (FiLM)**  
- Hypothesis: Order/Trade still contain incremental directional information once Market state is conditioned out.  
- Implementation: minimal FiLM or conditional norm on short event summaries; residual features only.  
- Kill: residual adds < 0.002 cosine after controlling for Market + v7, or collapses OOD (month-wise).  

---

# Final verdict

- **Most likely +0.001**: simple, well-calibrated magnitude gate on top of current best (lowest risk, direct exploitation of the |y| diagnostics and cosine mass).  
- **Most likely +0.003**: full MagGate-Innov (magnitude from Market + conditional directional innovation from O/T + cosine-optimal head). This is the single highest-expected-value architecture.  
- **Small but real chance of +0.01-scale breakthrough**: SyncScat (or SyncScat features inside MagGate) if the surrogate-identified geometric object really is still under-extracted by all previous representations. This is the highest-upside, highest-variance bet and the one that most cleanly escapes the “another sequence model / another residual” failure modes.

**Conditional Innovation is not rejected**; it is the right information-theoretic framing. The mistake to avoid is implementing it as “predict residual target” or “concat latents”. Implement it as **state-conditioned residualization of the event streams themselves**, keep everything short-range and relative, and optimize the head for the actual cosine decision rule rather than MSE.

Prioritize relative, local, phase/shape-aware, and conditionally normalized representations. That combination is the most coherent response to the entire set of empirical facts you have accumulated.















这是deepseek的方案# Executive Conclusion

**如果你只有3次大实验机会，跑这三个：**

| 优先级 | 方案 | 代号 | 预期增益 | 概率 |
|--------|------|------|----------|------|
| **1** | State-Conditioned Event Innovation + Conditional Normalization | **Project Surprise** | +0.003～0.005 | 最高 |
| **2** | Magnitude-Gated Directional Alpha with Copula Decomposition | **Project Magnitude** | +0.002～0.004 | 中高 |
| **3** | Time-Reversal Invariant Shapelet Network | **Project Shape** | +0.001～0.003（但有小概率+0.01） | 中 |

这三个方案的共同特征是：**没有一个是Transformer/TCN/LSTM的变体，没有一个是concat换皮，全部直接针对你发现的empirical facts设计。**

---

# Similar Competitions / Datasets

| Rank | Dataset / Competition | Market | Orders | Trades | Horizon | Target | Metric | Drift | Similarity |
|------|----------------------|--------|--------|--------|---------|--------|--------|-------|------------|
| 1 | **Optiver - Trading at the Close** | ✅ Nasdaq closing auction LOB | ✅ order book snapshots | ✅ executed trades | 10min | closing price movement | RMSE | high | ⭐⭐⭐⭐⭐ |
| 2 | **FI-2010** | ✅ 33 days, 1010 Finnish stocks | ✅ LOB 10 levels | ✅ transactions | 10-50 steps | mid-price direction | F1 | moderate | ⭐⭐⭐⭐ |
| 3 | **LOBSTER (NASDAQ)** | ✅ full LOB reconstruction | ✅ message-level | ✅ trades | variable | mid-price | various | moderate | ⭐⭐⭐⭐ |
| 4 | **Jane Street Real-Time Market Data** | ✅ anonymized market | ✅ 79 features | ❌ no explicit trades | 6 months | responder_6 | weighted R² | very high | ⭐⭐⭐ |
| 5 | **DeepLOB Benchmark** | ✅ LOB snapshots | ✅ 20 levels | ❌ | 25ms | mid-price | MSE/Accuracy | low | ⭐⭐⭐ |
| 6 | **Polymarket BTC 5-min** | ✅ 2-sec LOB | ✅ order book | ❌ | 5min | up/down | Accuracy | moderate | ⭐⭐⭐ |
| 7 | **JPX (Kaggle)** | ✅ Japanese stocks/options | ❌ | ❌ | future | return | R² | high | ⭐⭐ |
| 8 | **High Frequency Price Prediction (Kaggle)** | ✅ futures LOB | ✅ | ❌ | future | price movement | RMSE | unknown | ⭐⭐ |
| 9 | **CAC 40 futures (Euronext)** | ✅ 300 days LOB | ✅ | ✅ | variable | mid-price | various | moderate | ⭐⭐ |
| 10 | **Nasdaq TotalView-ITCH** | ✅ full message stream | ✅ add/cancel/exec | ✅ | N/A (generative) | N/A | N/A | N/A | ⭐⭐ |

**关键洞察**：MSCapital最独特的地方在于**三套数据的时间尺度嵌套结构**（600s market → 60s order → 60s transaction），这在公开数据集中几乎没有完全相同的。Optiver最接近但缺少market context的600s历史。因此，**方法迁移不能照搬，必须针对嵌套结构重新设计**。

---

# Literature Map

| Direction | Paper | Year | Core Idea | Source | Relevance |
|-----------|-------|------|-----------|--------|-----------|
| **LOB Foundation Models** | LOBERT | 2025 | BERT-style encoder for LOB messages, tokenizes multi-dim messages as single tokens | arXiv | ⭐⭐⭐ - message-level modeling思路可迁移 |
| **LOB Forecasting** | Deep Limit Order Book Forecasting (LOBFrame) | 2024 | Comprehensive benchmark of DL on NASDAQ LOB, releases open-source LOBFrame | arXiv | ⭐⭐⭐⭐ - 架构参考，特别是OOD分析 |
| **Transformer for LOB** | TLOB | 2025 | Dual attention for spatial+temporal dependencies on FI-2010 | arXiv/GitHub | ⭐⭐ - Transformer已试过，但dual attention思想可借鉴 |
| **Generative LOB** | TradeFM | 2026 | 524M-param generative Transformer on 9K equities, scale-invariant features | arXiv | ⭐⭐⭐ - scale-invariant tokenization值得学习 |
| **Return Decomposition** | CSM (Copula Sign-Magnitude) | 2026 | Decomposes returns into sign and magnitude via copula | ScienceDirect | ⭐⭐⭐⭐⭐ - **直接匹配你的target诊断** |
| **State-Dependent Hawkes** | ExsdHawkes | 2026 | Extends Hawkes with LOB physical constraints | arXiv | ⭐⭐⭐⭐ - state-conditioned event modeling |
| **Diffusion for LOB** | DiffVolume / DiffLOB | 2025-26 | Conditional diffusion for LOB volume generation | arXiv/ACM | ⭐⭐⭐ - conditional generation思路 |
| **Neural Hawkes** | Event-based LOB simulation | 2025 | Neural Hawkes for 12 LOB event types | Semantic Scholar | ⭐⭐⭐ - event innovation modeling |
| **MLP > Complex** | TLOB finding | 2025 | "Simple MLP-based architecture surpasses SoTA" | arXiv | ⭐⭐⭐⭐⭐ - **验证了你的RealMLP路线** |
| **Cryptocurrency LOB** | Microstructural Dynamics | 2025 | "Better inputs matter more than stacking layers" | arXiv | ⭐⭐⭐⭐ - 特征工程 > 模型复杂度 |

---

# Top 3 Candidate Architectures

## Candidate 1: Project Surprise —— State-Conditioned Event Innovation

### 1. 方法名称
**State-Conditioned Event Innovation Network (SCEIN)**

### 2. 来源
- 信息论链式分解：I(Y;M) + I(Y;O|M) + I(Y;T|M,O)
- Neural Hawkes Process / State-Dependent Hawkes
- Conditional Diffusion (DiffVolume/DiffLOB)
- FiLM / Conditional Normalization (computer vision迁移)

### 3. 为什么与MSCapital匹配

| Empirical Fact | How SCEIN Addresses It |
|----------------|------------------------|
| Market explains magnitude better than direction | Market → state encoder → predicts **expected** order/trade behavior (magnitude prior) |
| Order/Trade contain directional alpha | **Innovation** = observed − expected → captures direction conditional on market |
| Cross-channel sync is key | State conditions **all** channels simultaneously → synchronization = deviation from expected joint distribution |
| Time-reversal symmetric | Innovation is defined **locally** (≤30s), not directional |
| OOD risk | Conditional normalization is **sample-wise**, not global |

**核心公式**：
```
S = Encoder_Market(M)           # market state
μ_O, σ_O = Predictor(S)         # expected order flow distribution
Innov_O = (O - μ_O) / σ_O       # order innovation (surprise)
μ_T, σ_T = Predictor(S, O)      # expected trade distribution given S+O
Innov_T = (T - μ_T) / σ_T       # trade innovation
Y = Head(S, Innov_O, Innov_T)   # predict return
```

### 4. Architecture

```
Input:
  M: [B, 200, C_market]      # 600s market
  O: [B, L_order, C_order]   # 60s order events (variable length)
  T: [B, L_trade, C_trade]   # 60s trade events (variable length)

Step 1 - Market State Encoder:
  M → LocalConv1D(kernel=3-10, groups=C) → LayerNorm → GELU
     → CrossChannelMixing (1x1 conv) → GlobalAvgPool → S [B, d_state]
  # 设计原则：短程、跨通道、时间反转不敏感

Step 2 - Conditional Predictors (FiLM-style):
  For Order:
    S → FiLM_generator → γ_O, β_O
    O_norm = LayerNorm(O)
    O_cond = γ_O * O_norm + β_O
    O → small predictor → μ_O, σ_O (per timestep)
    Innov_O = (O - μ_O) / (σ_O + ε)
  
  For Trade (conditioned on S + pooled O):
    [S, Pool(O)] → FiLM_generator → γ_T, β_T
    T_norm = LayerNorm(T)
    T_cond = γ_T * T_norm + β_T
    T → predictor → μ_T, σ_T
    Innov_T = (T - μ_T) / (σ_T + ε)

Step 3 - Fusion & Head:
  [S, Pool(Innov_O), Pool(Innov_T)] → MLP (2-3 layers) → y_pred

Loss:
  Cosine loss (primary) + auxiliary: MSE(μ_O, O) + MSE(μ_T, T)
  # 辅助loss迫使predictor真正学习“expected” behavior
```

### 5. 为什么以前的方法没有覆盖它

- **不是concat**：O和T不是直接concat到M，而是通过**条件化**引入
- **不是Transformer**：没有self-attention，用的是conditional normalization + local conv
- **不是TCN**：encoder只用了短程conv（≤10 timestep），符合有效尺度≤30s的发现
- **不是residual**：Innovation不是y-baseline的residual，而是**modality内部的surprise**
- **不是feature engineering**：Innovation是**学习的**，不是手工构造的

### 6. 预期新增信息

| Term | What it captures |
|------|------------------|
| I(Y;M) | Market → magnitude prior (已部分exploit) |
| **I(Y;O\|M)** | **Order surprise conditional on market** (全新) |
| **I(Y;T\|M,O)** | **Trade surprise conditional on market+order** (全新) |

### 7. 最大风险

1. **OOD**：Conditional predictor可能过拟合到train的市场状态分布
2. **Implementation complexity**：order和trade是variable-length event streams，对齐困难
3. **Auxiliary loss balancing**：可能dominate cosine loss

### 8. 最小验证实验（2-6小时）

**Probe**: 用简单的线性模型验证“innovation”是否有信息增量

```
Step 1: 用market → 训练一个小MLP预测O的每个channel（MSE）
Step 2: 计算Innov_O = O - O_predicted
Step 3: 训练三个模型：
  A: market → y
  B: market + O → y
  C: market + Innov_O → y
比较: corr(C, y) vs corr(B, y)
```

**如果** `corr(C, y) >= corr(B, y) * 0.9`，说明Innovation保留了O的信息且更紧凑 → **继续**。
**如果** `corr(C, y) < corr(B, y) * 0.5`，说明预测器太差或innovation不保留信息 → **KILL**。

### 9. Kill Criteria

| Metric | Threshold | Action |
|--------|-----------|--------|
| corr(Innov_O, O) | < 0.3 | KILL（predictor没学到东西） |
| Δcosine (v7 + Innov_O) | < +0.0005 | KILL（信息增量不足） |
| Monthly consistency | < 50% months positive | KILL（不稳定） |

### 10. 成功后的完整版

1. 用更复杂的encoder（LocalConv + ChannelMixing）
2. 用Transformer对variable-length O/T做pooling
3. 端到端训练（不再两阶段）
4. Ensemble with v7

---

## Candidate 2: Project Magnitude —— Magnitude-Gated Directional Alpha

### 2.1 方法名称
**Copula Sign-Magnitude Decomposition Network (CSM-Net)**

### 2.2 来源
- Christoffersen et al. (2026) “A new decomposition approach to modeling financial returns: Conditioning sign on magnitude”
- 你的核心发现：`corr(market, |y|) ≈ 0.43-0.47`，`corr(|v7|, |y|) ≈ 0.156`
- 理论：`P(sign | magnitude, x)` 可能比 `P(sign | x)` 更强

### 2.3 为什么与MSCapital匹配

| Empirical Fact | How CSM-Net Addresses It |
|----------------|--------------------------|
| Market predicts magnitude far better than direction | Market → dedicated **magnitude predictor** |
| v7在大波动时方向最准（0.585 vs 0.37） | **Condition sign on magnitude**：大波动时用v7的方向，小波动时shrink |
| Cosine metric重视大波动样本（top 5%贡献56.5%） | Magnitude-gated prediction直接优化cosine的样本权重结构 |
| Market的magnitude信息未被充分利用 | Explicit magnitude modeling |

**核心数学推导**：

对于uncentered cosine similarity，最优预测是什么？

```
Metric = Σ(pred_i * y_i) / sqrt(Σ pred_i² * Σ y_i²)

在L2 norm约束下（||pred|| = 1），最大化内积：
  optimal pred ∝ y / ||y||  （在样本维度上）
  
但这不是E[y|x]，而是E[y/||y|| | x]的某种近似。

更精确地：
  如果pred = α * sign_hat * mag_hat，
  则cosine ≈ (α * Σ sign_hat * mag_hat * y_i) / (α * sqrt(Σ mag_hat²) * sqrt(Σ y_i²))
             = Σ sign_hat * (mag_hat/||mag_hat||) * y_i / ||y||
  
  最优策略：用magnitude预测对样本做**soft weighting**，
  方向预测在大magnitude样本上权重更高。
```

因此最优预测不是E[y|x]，而是：

```
pred* = argmax_pred E[ cosine(pred, y) | x ]
      ≈ E[ sign(y) * |y| / E[|y|²]^{1/2} | x ]
```

这意味着：**应该联合建模sign和magnitude，且magnitude影响prediction的scale。**

### 2.4 Architecture

```
Input: M, O, T (same as before)

Branch 1 - Magnitude Predictor:
  [M, O, T] → encoder → MLP → mag_hat ∈ [0, ∞)
  Loss: MSE(log mag_hat, log |y|)  # log空间稳定

Branch 2 - Direction Predictor (conditioned on magnitude):
  [M, O, T, mag_hat] → encoder → MLP → sign_logits
  # 关键：magnitude作为**条件特征**输入
  Loss: Binary cross-entropy (weighted by mag_hat)
        # 大magnitude样本权重更高

Branch 3 - Joint Head:
  sign_prob = sigmoid(sign_logits)
  sign_hat = 2 * sign_prob - 1
  y_pred = sign_hat * mag_hat * scale_factor
  
  # scale_factor是可学习的，用于匹配cosine的最优scale

Loss:
  L = -cosine(y_pred, y) + λ₁ * MSE(log mag_hat, log |y|) + λ₂ * BCE_weighted
```

**关键设计**：Direction branch的输入包含`mag_hat`，实现了`P(sign | magnitude, x)`的建模。

### 2.5 为什么以前的方法没有覆盖它

- **不是简单multiplication**：magnitude是**条件变量**输入到direction branch
- **不是两阶段**：端到端训练，gradient从cosine回传到两个branch
- **不是residual**：完全不同的分解方式
- **不是feature engineering**：学习的是**预测策略**，不是手工特征

### 2.6 预期新增信息

| Term | What it captures |
|------|------------------|
| I(Y;M) | Market → magnitude (已部分exploit) |
| **I(sign; M,O,T \| magnitude)** | **Direction conditional on magnitude** (全新) |
| **Joint distribution P(sign, magnitude)** | **Copula interaction** (全新) |

### 2.7 最大风险

1. **Magnitude预测过拟合**：market的magnitude signal可能OOD不稳定
2. **Loss balancing**：三个loss的权重需要仔细调
3. **Scale factor可能爆炸**

### 2.8 最小验证实验（2-6小时）

**Probe**: 验证conditional sign prediction是否比unconditional更强

```
Step 1: 用|y|将样本分成5个quantile bins
Step 2: 在每个bin内，计算：
  - v7的方向准确率
  - market-only的方向准确率
Step 3: 训练两个简单模型：
  A: M+O+T → sign (unconditional)
  B: M+O+T+|y|_pred → sign (conditional)
比较: 在最大|y| bin中，B的准确率是否显著高于A
```

**如果**在top quantile中B的准确率 > A + 0.03 → **继续**。
**如果**没有显著差异 → **KILL**（conditional sign没有额外信息）。

### 2.9 Kill Criteria

| Metric | Threshold | Action |
|--------|-----------|--------|
| corr(mag_hat, |y|) | < 0.3 | KILL |
| Top-quantile direction accuracy | < 0.55 | KILL |
| Δcosine | < +0.0005 | KILL |

### 2.10 成功后的完整版

1. 用copula显式建模P(sign, magnitude)的联合分布
2. 多horizon预测（多个future horizons）
3. 用normalizing flow做更灵活的分布建模

---

## Candidate 3: Project Shape —— Time-Reversal Invariant Shapelet Network

### 3.1 方法名称
**Time-Reversal Invariant Shapelet Network (TRIS-Net)**

### 3.2 来源
- **Shapelet Learning** (时间序列分类的经典方法，从UCR archive迁移)
- **Scattering Transform** (信号处理，Mallat, 2012)
- **Time-Reversal Invariant Networks** (物理ML，从粒子物理迁移)
- 你的surrogate实验：reverse sequence几乎不下降，block shuffle >10 steps后性能下降

### 3.3 为什么与MSCapital匹配

| Empirical Fact | How TRIS-Net Addresses It |
|----------------|---------------------------|
| Reverse sequence几乎不下降 | **Explicit time-reversal invariance**: f(x) + f(reverse(x)) |
| Block shuffle >10 steps后下降 | **Local shapelets** (≤10 timestep) |
| Cross-channel sync是关键 | Shapelets are **multivariate** — capture cross-channel patterns |
| Nonlinear phase敏感 | Shapelet distance is **phase-aware** but reversal-invariant |
| OOD risk | Shapelets are **local patterns**, not global regimes |

**核心洞察**：Market signal ≈ 短时间尺度内多通道同步发生的**非线性形态**。这本质上是一个**multivariate time series shape matching**问题，而不是sequence modeling问题。

### 3.4 Architecture

```
Input: M [B, 200, C]

Step 1 - Learnable Shapelets:
  K个shapelets，每个shapelet: [L, C]，其中L ∈ {3, 5, 7, 10} (多尺度)
  # 每个shapelet是一个"模板"形态

Step 2 - Shapelet Distance:
  For each shapelet k (length L):
    For each position t in M:
      dist_k(t) = ||M[t:t+L] - shapelet_k||²
    # 或使用更robust的cosine distance
  
  # 关键：同时计算 forward 和 reverse 的距离
  dist_k_fwd(t) = ||M[t:t+L] - shapelet_k||²
  dist_k_rev(t) = ||reverse(M[t:t+L]) - shapelet_k||²
  dist_k(t) = min(dist_k_fwd(t), dist_k_rev(t))  # time-reversal invariant!

Step 3 - Soft-min Pooling:
  For each shapelet k:
    s_k = -log(Σ_t exp(-dist_k(t) / τ))  # 最近似的位置
  # 或者：top-k最小距离的平均

Step 4 - Cross-Shapelet Interaction:
  S = [s₁, s₂, ..., s_K] → MLP → y_pred

Optional - Shapelet Visualization:
  将学习到的shapelet可视化，检查是否学到有意义的微观结构形态
```

### 3.5 为什么以前的方法没有覆盖它

- **不是TCN**：没有causal convolution，没有时间方向性
- **不是Transformer**：没有attention over time，用的是**template matching**
- **不是CNN**：不是hierarchical feature extraction，是**explicit shape matching**
- **不是feature engineering**：shapelets是**学习的**
- **完全匹配time-reversal symmetric的发现**

### 3.6 预期新增信息

| Term | What it captures |
|------|------------------|
| **I(Y;M)** | **通过shape matching重新提取**，与之前所有方法正交 |

关键：TRIS-Net提取的information与v7的corr可能很低（<0.3），因为v7用的是152 event-dynamics features + RealMLP，完全是不同的归纳偏置。

### 3.7 最大风险

1. **Shapelet learning不稳定**：可能学到噪声pattern
2. **计算复杂度**：O(B * T * L * K * C) 可能较大
3. **可解释性vs预测性**：shapelets可能可解释但不预测

### 3.8 最小验证实验（2-6小时）

**Probe**: 用随机shapelets验证shape matching是否有信息

```
Step 1: 随机初始化K个shapelets（不训练）
Step 2: 对每个样本计算到每个shapelet的距离
Step 3: 用这些距离作为特征训练一个简单MLP → y
Step 4: 与random baseline比较
```

**如果** random shapelets + MLP 能达到 > 0.05 cosine → 说明shape matching方向有信号 → **继续**。
**如果** < 0.02 → **KILL**（shape matching本身不捕获信息）。

**进阶Probe**: 训练shapelets（端到端）看是否超过random。

### 3.9 Kill Criteria

| Metric | Threshold | Action |
|--------|-----------|--------|
| Random shapelet baseline | < 0.03 | KILL |
| Trained shapelet Δcosine | < +0.0005 | KILL |
| Shapelet diversity | < 0.5 pairwise distance | KILL（都在学相同pattern） |

### 3.10 成功后的完整版

1. 多尺度shapelets (L=3,5,7,10)
2. Hierarchical shapelets (组合成更复杂的形态)
3. Shapelet + v7 ensemble (预期相关性低，ensemble增益大)

---

# 最值得先跑的实验

## Experiment 1: Innovation Probe

| | |
|---|---|
| **Hypothesis** | Order flow conditional on market state contains information not captured by market alone; innovation (O - E[O\|M]) preserves this information |
| **Implementation** | 两阶段：① MLP: M → O (MSE) ② 比较 v7 vs v7 + Innov_O |
| **Baseline** | v7 (0.142) |
| **Metric** | Public LB cosine |
| **Expected result** | +0.001～0.002 |
| **Kill criterion** | Δcosine < +0.0005 |
| **Next step** | 如果成功 → 扩展到Trade innovation + 端到端训练 |

## Experiment 2: Magnitude-Gated Direction

| | |
|---|---|
| **Hypothesis** | Conditioning direction prediction on estimated magnitude improves direction accuracy in high-|y| samples, which dominate cosine metric |
| **Implementation** | 两个简单模型：① M→\|y\| ② [M, \|y\|_hat]→sign，用|y|加权训练 |
| **Baseline** | v7 |
| **Metric** | Top-quantile direction accuracy, cosine |
| **Expected result** | Top-quantile accuracy +0.03～0.05 |
| **Kill criterion** | Top-quantile accuracy < 0.55 |
| **Next step** | 如果成功 → 端到端CSM-Net + copula |

## Experiment 3: Shapelet Random Probe

| | |
|---|---|
| **Hypothesis** | Market signal is in local cross-channel shapes, not temporal dynamics; shape matching can extract orthogonal information |
| **Implementation** | 随机shapelets + MLP (不训练shapelets，只训练MLP) |
| **Baseline** | Random features baseline |
| **Metric** | cosine correlation |
| **Expected result** | > 0.05 cosine (证明shape space有信号) |
| **Kill criterion** | < 0.02 cosine |
| **Next step** | 如果成功 → 端到端shapelet learning |

---

# Final Verdict

| 方案 | 最可能增益 | 理由 |
|------|-----------|------|
| **Project Surprise (SCEIN)** | **+0.003～0.005** | 直接利用未使用的conditional information I(Y;O\|M)和I(Y;T\|M,O)，理论最solid，与你的信息论分析完全一致 |
| **Project Magnitude (CSM-Net)** | **+0.002～0.004** | 直接利用你已发现的market→magnitude signal和cosine的样本权重结构，数学上有严格推导 |
| **Project Shape (TRIS-Net)** | **+0.001～0.003**（小概率+0.01） | 最反直觉，与所有已有方法正交，如果shapelet真的捕获了cross-channel synchronization，ensemble增益可能很大 |

**最终建议执行顺序**：

1. **先跑Experiment 1 (Innovation Probe)** —— 最高概率成功，2-6小时出结果
2. **同时跑Experiment 3 (Shapelet Random Probe)** —— 低成本验证反直觉方向
3. **如果1成功 → 完整SCEIN；如果3有信号 → 完整TRIS-Net；然后跑Experiment 2**

**保守估计**：三个方案中至少1-2个能带来可测量增益，叠加后可能冲击 **0.148-0.150**。



这是GPT2的方案：
# Executive conclusion

我把你给出的 MSCapital 诊断、截至 **2026-08-14** 能找到的相似比赛、LOB benchmark、2024–2026 新论文和能实际读到的开源实现交叉筛了一遍。

先给结论：

> **如果现在只有 3 次真正值得烧计算资源的大实验机会，我会按下面顺序跑：**
>
> **① MAG-MoE：Magnitude-Conditioned Conditional-Mean Mixture**
>
> Market 不负责直接给方向，而负责判断“这次状态下哪一种 Order/Trade alpha 值钱”，用 magnitude/extreme probability 做 **conditional expert gating**。
>
> **② SCFI：State-Conditioned Flow Innovation**
>
> 不预测 `y-v7`，而是预测“在当前 Market state 下本来应该出现什么 Order/Trade”，再取 **event innovation / point-process compensator**。这是我认为真正可能出现大台阶的路线。
>
> **③ RICS：Reversal-Invariant Cross-Channel Synchronization**
>
> 不上 Transformer/TCN，而是显式编码 **≤30s 跨通道 phase synchronization / cross-spectrum / local covariance**，并强制 time-reversal invariance。这是反直觉方案。

我现在认为：

| 目标               | 最可能方案                                    |
| ---------------- | ---------------------------------------- |
| **稳定 +0.001 左右** | **RICS**                                 |
| **有希望 +0.003**   | **MAG-MoE**                              |
| **小概率真正 +0.01**  | **SCFI**                                 |
| 冲击 Public 0.15+  | **只有 SCFI 的信息结构让我认为存在这个 ceiling；但概率并不高** |

原因不是“这些模型更新”，而是三个方案分别利用了目前没被充分兑现的三个东西：

[
\boxed{
\text{Market magnitude}
\rightarrow
\text{conditional usefulness of event alpha}
}
]

[
\boxed{
I(Y;O|M),\quad I(Y;T|M,O)
}
]

以及

[
\boxed{
\text{local cross-channel nonlinear phase structure}
}
]

这正对应你已经定位出来的 Market 信息：短尺度、跨通道同步、phase-sensitive、近似 time-reversal symmetric。

---

# 一、我对“哪个比赛最像 MSCapital”的结论

先说一个重要判断：

## **最像 MSCapital 的不是某一个 Kaggle 比赛。**

而是三个领域的交集：

> **LOBSTER / Nasdaq message-level LOB**
> +
> **Optiver Realized Volatility 的 magnitude forecasting**
> +
> **state-dependent event-flow / Hawkes 文献**

MSCapital 特别之处在于同时给你：

* 600s coarse market state；
* 最后 60s raw order event；
* 最后 60s transaction event；
* future return；
* cosine objective。

这比标准 FI-2010 / DeepLOB benchmark 信息层次更丰富。

## Top 10 相似数据集 / 比赛

这里的“相似度”是我按**数据生成结构**给出的研究排序，不是官方指标。

| Rank   | Dataset / Competition                          |               Market |               Orders |        Trades | Target                 | Drift / Validation  |       相似度 |
| ------ | ---------------------------------------------- | -------------------: | -------------------: | ------------: | ---------------------- | ------------------- | --------: |
| **1**  | **LOBSTER / Nasdaq TotalView-ITCH**            |                ✅ LOB |       ✅ raw messages |  ✅ executions | 可构造 future return      | 可跨日/跨年              |    **96** |
| **2**  | **Optiver Realized Volatility**                |              ✅ L1/L2 |       △ book updates | ✅ trade table | future volatility      | 真实未来 3 个月           |    **91** |
| **3**  | **2026 Binance L2 + trade-flow state dataset** |          ✅ Top-20 L2 |               △ flow |             ✅ | future liquidity state | rolling monthly OOS |    **90** |
| **4**  | **LOBFrame Nasdaq 2017–2019**                  |                ✅ LOB |                    △ |             △ | future mid-price       | 强 temporal split    |    **87** |
| **5**  | **TLOB TSLA–INTC benchmark**                   |       ✅ 10-level LOB | ✅ message/order info |             △ | future direction       | chronological       |    **85** |
| **6**  | **Digital Asset LOB / Coinbase**               |                ✅ LOB |                    △ |             △ | 2s future move         | walk-forward        |    **81** |
| **7**  | **FI-2010**                                    |           ✅ 10-level |         ❌ raw order弱 |             ❌ | mid-price class        | 10 days             |    **74** |
| **8**  | **WSELOB-2017**                                |           ✅ full LOB |        ✅ raw updates |             △ | 可构造                    | 整整 1 年              |    **73** |
| **9**  | **OpenMarket 2026**                            | ✅ synchronized books |               ✅ flow |             ✅ | cross-venue response   | walk-forward        |    **71** |
| **10** | **Optiver Trading at the Close / DRW Crypto**  |       ✅ market state |                    △ |             △ | future price           | strong OOS          | **63–67** |

### 1. LOBSTER 为什么第一

LOBSTER 是我认为**数据生成机制最接近 MSCapital**的公开数据体系。

它不是普通 K 线数据，而是从 Nasdaq Historical TotalView-ITCH message feed 重建 order book；因此 book state 和 event messages 来自同一个真实撮合过程。([Lobster Data][1])

也就是说，可以天然构造：

[
MarketState_t
\rightarrow
OrderEvents_{t:t+\Delta}
\rightarrow
Executions
\rightarrow
PriceMove
]

这和 MSCapital 的

[
M\rightarrow O\rightarrow T\rightarrow Y
]

最接近。

**后续继续找方法时，我建议把 LOBSTER 方法论文的权重提高，而不是继续把 Optiver 当唯一母比赛。**

---

## 2. Optiver Realized Volatility 为什么排第二

它没有 MSCapital 那么完整的 raw order event，但它有一个极其重要的相似性：

> **book + trade → future magnitude**

Optiver RV 使用短时间 order-book/trade 信息预测后续 10-minute realized volatility，并且比赛真正面对了未来市场数据的时间外验证。([Kaggle][2])

更重要的是冠军方案并不是“一个巨型时序 Transformer”。

第一名最终组合了：

* LightGBM
* 1D-CNN
* MLP

而且冠军帖直接强调 nearest-neighbor / market structure 思路。([Kaggle][3])

这和你的 Market 诊断非常值得连起来：

[
corr(Market, |y|)
\approx0.43\sim0.47
]

远强于 Market 对 sign 的能力。

**所以 Optiver RV 真正值得迁移的不是模型，而是“magnitude 是独立预测问题”这一分解。**

---

## 3. 2026 年这篇论文非常关键

我认为这可能是这轮调研最与你比赛契合的新论文：

### *When Does Order Flow Matter? State-Dependent L2 Liquidity-State Transitions in Crypto Futures*

2026 年 7 月。

数据直接包含：

* Binance BTC/ETH futures；
* top-20 L2；
* trade-flow；
* 2023–2026；
* rolling monthly OOS；
* event-clustered validation；
* blocked permutation tests。([arXiv][4])

它得到一个对我们非常重要的结果：

> **第一层最重要的是 pre-event L2 state；order flow 不能替代 L2 state，只有 layered on top of L2 state 后才出现额外价值。** ([arXiv][4])

翻译成 MSCapital：

[
\boxed{
O\text{ 的意义取决于 }M
}
]

而不是

[
M\ encoder \oplus O\ encoder
]

这实际上就是：

[
I(Y;O|M)
]

的经验版本。

---

# 二、Literature map：哪些论文真正值得读源码

我把“论文很 SOTA，但和你数据诊断不匹配”的过滤掉了。

| 方向                            | 方法                                 |      年份 | 真正值得迁移的部分                                    |              源码 |   MSCapital 价值 |
| ----------------------------- | ---------------------------------- | ------: | -------------------------------------------- | --------------: | -------------: |
| State-conditioned events      | **sd-PNHP**                        |    2022 | market state conditioned event intensity     |               ✅ |          ⭐⭐⭐⭐⭐ |
| OFI conditional dynamics      | **Forecasting High Frequency OFI** |    2024 | Hawkes 预测 expected order flow                |               — |          ⭐⭐⭐⭐⭐ |
| State-first microstructure    | **When Does Order Flow Matter?**   |    2026 | state → incremental flow                     |               — |          ⭐⭐⭐⭐⭐ |
| Structured MLP                | **MLPLOB/TLOB**                    |    2025 | feature/time mixing + BiN                    |               ✅ |           ⭐⭐⭐⭐ |
| Higher-order channel relation | **HLOB**                           |    2024 | MI structure among book variables            |      ✅ LOBFrame |           ⭐⭐⭐⭐ |
| Event density                 | **LOBDIF**                         | 2024/26 | joint time/event conditional density         |               ✅ |            ⭐⭐⭐ |
| Representation                | **LOBench**                        |    2025 | transferable representation benchmark        |               ✅ |            ⭐⭐⭐ |
| Message model                 | **LOBERT**                         |    2025 | continuous price/volume/time tokens          |               — |             ⭐⭐ |
| Local covariance              | **STVNN**                          |    2024 | covariance-based relational bias + stability |               — | ⭐⭐⭐⭐⭐ for RICS |
| Local shape                   | **ShapeFormer / ShapeNet**         | 2021–24 | discriminative subsequences                  |               ✅ |            ⭐⭐⭐ |
| Signal processing             | **Joint TF Scattering**            | classic | local multiscale TF structure                | implementations |             ⭐⭐ |

sd-PNHP 有官方实现；LOBFrame、TLOB、LOBDIF、DAIN、LOBench 也都有公开代码。([GitHub][5])

---

# 三、源码调查后，一个非常有价值的发现

## sd-PNHP 和你的数据非常像

它的公开代码不是概念 demo。

实际配置是：

```text
event sequence length = 50
event types = 4
hidden_dim = 16
embedding_dim = 16
batch = 512
optimizer = RMSprop
```

数据按时间记录切：

```text
records 1–3 → train
record 4   → validation
record 5   → test
```

([GitHub][6])

更重要的是 architecture。

它分别 embedding：

[
event\ type
]

和

[
market\ state
]

然后：

[
x_t=
Embedding(Event_t)
+
Embedding(State_t)
]

再进入 continuous-time LSTM。代码里 market state 是显式条件，而不是最后 concat。([GitHub][7])

最终学习：

[
\lambda_k(t|H_t,S_t)
]

即不同 event type 的 conditional intensity。

训练目标是真正的 Hawkes negative log likelihood，同时代码也可以加入 next-event classification loss。([GitHub][7])

---

# 四、这里就是 Conditional Innovation 的成熟数学对应

对于第 (k) 类 Order event：

设条件强度：

[
\lambda_k(t|M)
]

那么在窗口 (W) 内：

[
\hat\Lambda_k
=============

\int_W \lambda_k(t|M),dt
]

实际事件数量为：

[
N_k(W)
]

于是可以构造：

[
\boxed{
Innovation_k
============

N_k(W)-\hat\Lambda_k(W)
}
]

更稳一点：

[
\boxed{
Z_k=
\frac{N_k-\hat\Lambda_k}
{\sqrt{\hat\Lambda_k+\epsilon}}
}
]

这不是：

[
y-\hat y
]

而是：

[
\boxed{
Observed\ OrderFlow
-------------------

Expected\ OrderFlow|Market
}
]

因此**完全绕开你已经验证失败至少 5 次的 target residual modeling**。

这就是我认为 SCFI 值得保留的根本原因。

---

# 五、Cosine metric 的严格推导

这部分我认为也非常关键。

假设大量 iid/近似 ergodic samples，模型输出：

[
p=f(X)
]

测试 cosine 收敛到：

[
C(f)
====

\frac{E[f(X)Y]}
{\sqrt{E[f(X)^2]E[Y^2]}}
]

因为：

[
m(X)=E[Y|X]
]

所以：

[
E[f(X)Y]
========

E[f(X)m(X)]
]

Cauchy-Schwarz：

[
E[fm]
\le
\sqrt{E[f^2]E[m^2]}
]

等号成立当：

[
f(X)=c,m(X),\qquad c>0
]

因此：

[
\boxed{
f^*(X)\propto E[Y|X]
}
]

## 所以一个重要结论是：

**Cosine 的 Bayes-optimal output 并没有变成单纯 sign。**

也不是：

[
E[\operatorname{sign}(Y)|X]
]

也不是随意：

[
P(|Y|>P90|X)\times direction
]

---

## 更严格地说

有限 test vector 下：

[
C(p,Y)
======

\frac{p^\top Y}{|p||Y|}
]

给定所有 test features (X_{1:n})：

[
E[C|X]
======

\frac{
p^\top
E[Y/|Y||X]
}{
|p|
}
]

所以精确最优：

[
\boxed{
p^*
\parallel
E\left[
\frac{Y}{|Y|}
\mid X_{1:n}
\right]
}
]

但 (n) 大时 (|Y|) 集中：

[
p_i^*\approx E[Y_i|X_i].
]

---

# 六、为什么 magnitude 仍然极其有用

把：

[
Y=S\cdot A
]

其中：

[
S\in{-1,+1},\quad A=|Y|
]

则：

[
E[Y|X]
======

## P(S=+|X)E[A|S=+,X]

P(S=-|X)E[A|S=-,X]
]

所以真正应该建模的是：

[
\boxed{
P(sign|X)
+
E[|Y||sign,X]
}
]

如果额外满足条件独立：

[
A\perp S|X
]

才可以简化成：

[
E[A|X]
\cdot
(2P(S=+|X)-1)
]

因此：

## **简单 `magnitude × v7` 没有足够数学依据。**

但是：

## **Magnitude-conditioned experts 有。**

这就是 Candidate #1。

---

# 七、Top 3 candidate architectures

# 🥇 Candidate 1 — MAG-MoE

## Magnitude-Conditioned Conditional-Mean Mixture

研究代号：

[
\boxed{\text{MAG-MoE}}
]

### 为什么排名第一

因为你已经有非常强的本地证据：

Market：

[
corr(M,|y|)=0.43\sim0.47
]

而：

[
corr(|v7|,|y|)\approx0.156
]

同时：

* largest-|y| 上 v7 direction accuracy ≈ 0.585；
* cosine top 20% 样本贡献约 74.4%；
* extreme 5% 贡献约 56.5%。

这是几乎为 gating 量身定做的结构。

---

## Architecture

不要：

```text
market magnitude
      ×
v7 prediction
```

改成：

```text
Market local/state encoder
        │
        ├─ q70 = P(|y| > P70)
        ├─ q90 = P(|y| > P90)
        ├─ q97 = P(|y| > P97)
        └─ magnitude embedding
                ↓
          low-capacity gate
                ↓
     ┌──────────┼──────────┐
 normal expert  move expert  extreme expert
     │            │             │
  Order/Trade  Order/Trade   Order/Trade
   RealMLP      RealMLP       RealMLP
     └──────────┼─────────────┘
                ↓
           weighted sum
                ↓
              pred
```

Gate：

[
g=
softmax(Wq+b)
]

Prediction：

[
\hat y=
\sum_{k=1}^{K}g_k(M),f_k(O,T)
]

重点是：

> **Market 决定哪个 O/T predictor 应被信任。**

而不是把 Market 自己直接当 prediction feature。

---

## 新利用了什么信息

不是单纯：

[
I(Y;M)
]

而是：

[
\boxed{
\text{interaction / synergy between }M\text{ and }(O,T)
}
]

也就是：

[
P(Y|O,T,M)
]

里：

[
M
]

改变

[
O,T\rightarrow Y
]

这条 mapping。

最新 2026 state-first 结果正好说明 order-flow incremental value 会依赖当前 liquidity state。([arXiv][4])

---

## 为什么不是 concat 换皮

Concat：

[
f(M,O,T)
]

要求网络自己发现 conditional routing。

MAG-MoE 强制：

[
\boxed{
M\rightarrow Gate
}
]

[
\boxed{
O,T\rightarrow Experts
}
]

结构上明确施加：

> Market 是 context，而 event flow 是 alpha source。

---

## 最大风险

gate 很容易学成：

```text
大 volatility
→ prediction amplitude 放大
```

这会变成 disguised multiplication。

所以必须检查：

[
Var(g_k)>0
]

并且不同 expert 的 event-feature response 真正不同。

---

# 🥈 Candidate 2 — SCFI

# State-Conditioned Flow Innovation

研究代号：

[
\boxed{\text{SCFI}}
]

我认为这是**真正突破 ceiling 最大**的。

---

## Phase 1：Market → Expected Order

取最后 30s 或多尺度：

```text
Market
[10 timestep × C]
      ↓
relative/sample normalization
      ↓
small state encoder
      ↓
conditional Order distribution
```

Order 不需要一上来预测完整 event sequence。

第一版只做：

每 5s / 10s：

* bid add count；
* ask add count；
* bid cancel count；
* ask cancel count；
* signed size；
* OFI；
* mean distance；
* inter-event-time；
* size quantiles。

约：

[
D_O=32\sim64
]

预测：

[
\mu_O(M),\quad\sigma_O(M)
]

然后：

[
Z_O=
\frac{
O-\mu_O(M)
}{
\sigma_O(M)+\epsilon
}
]

---

## Phase 2：Market + Order → Expected Trade

再：

[
(M,Z_O)
\rightarrow
\mu_T,\sigma_T
]

得到：

[
Z_T=
\frac{
T-\mu_T(M,O)
}{
\sigma_T(M,O)+\epsilon
}
]

---

## Phase 3

最终：

```text
Order Innovation
+
Trade Innovation
      ↓
small RealMLP
      ↓
prediction
```

先不要加全部 152。

否则很容易又被原 feature space 吞掉。

---

## 完整版

如果 probe 成功，再升级成：

[
\lambda_O(t,k|M)
]

以及：

[
\lambda_T(t,k|M,O)
]

用 point-process compensator：

[
dN_t-\lambda_tdt
]

作为 representation。

这恰好借用了 sd-PNHP 的成熟机制，但**Hawkes 不负责预测 y**。

Hawkes 只负责回答：

> “现在这种 Market state 下，这个 Order event 到底意不意外？”

这比直接拿 Hawkes 做价格预测合理得多。2024 的 OFI 研究也证明 Hawkes 可以用于预测 near-term OFI distribution，而不是必须直接预测价格。([arXiv][8])

---

## 它新增的随机变量

Candidate #2 是三个方案里回答这个问题最漂亮的：

[
\boxed{I(Y;O|M)}
]

和：

[
\boxed{I(Y;T|M,O)}
]

---

## 为什么可能抗 OOD

普通 sequence net 学：

[
AbsolutePattern\rightarrow Y
]

SCFI 学：

[
ObservedBehavior-
ExpectedBehaviorUnderCurrentState
]

如果：

```text
month 20：
低流动性 → cancel count 200

month 90：
低流动性 → cancel count 500
```

absolute count 已经 drift。

但：

[
\frac{Observed-Expected}{ExpectedScale}
]

可能仍然表达同一件事：

> “cancel activity 比此状态下正常水平异常高。”

这是它最大的 OOD 理论优势。

---

## 风险

**conditional expectation model 本身也会漂移。**

所以 nuisance model：

[
M\rightarrow O
]

必须：

* 极低 capacity；
* relative inputs；
* rank / scale normalization；
* chronological CV；
* 最好 cross-fitting。

不要用大 Transformer。

---

# 🥉 Candidate 3 — RICS / PHASELOCK

# Reversal-Invariant Cross-Channel Synchronization Encoder

这是我要选的**反直觉方案**。

来源不是金融主流 Transformer，而是：

* signal processing；
* neuroscience phase synchronization；
* spatiotemporal covariance；
* multivariate shape analysis。

---

## 为什么我反而不推荐 plain scattering

Joint time-frequency scattering 确实擅长稳定、多尺度、shift-invariant 的 time-frequency representation。([arXiv][9])

但你的 surrogate 已经明确证明：

> **phase destruction 几乎毁掉全部 Market signal。** 

而 scattering 的 modulus/invariance 有可能把我们最珍贵的**relative phase**也削弱。

所以：

### 不跑 plain scattering。

跑：

# **cross-channel phase coherence**

---

## Architecture

只取：

[
X\in\mathbb R^{10\times C}
]

也就是最后约 30 秒。

先做每 sample：

```text
mid-relative price
spread relative
depth ratio
volume log/rank
zero-centered channel
```

然后 local rFFT：

[
Z_c(f)=FFT(X_c)
]

通道 (i,j)：

[
S_{ij}(f)=
Z_i(f)Z_j(f)^*
]

normalized coherence：

[
C_{ij}(f)=
\frac{
S_{ij}(f)
}{
\sqrt{
S_{ii}(f)S_{jj}(f)
}+\epsilon
}
]

保存：

[
Re(C_{ij})
]

和：

[
|C_{ij}|
]

---

## 为什么这和 reverse invariance 特别匹配

time reverse 会让 Fourier phase conjugate。

因此：

[
C_{ij}
\rightarrow C_{ij}^{*}
]

于是：

[
Re(C_{ij})
]

不变。

[
|C_{ij}|
]

也不变。

也就是说它天然编码：

[
\boxed{
phase\ synchronization
}
]

同时满足：

[
\boxed{
time\ reversal\ invariance
}
]

这几乎精确对应你的 surrogate result。

---

## 再加 local covariance

[
\Sigma=
\frac1T X^\top X
]

以及 lag covariance：

[
\Gamma_k
========

X_{1:T-k}^{\top}X_{1+k:T}
]

使用：

[
\Gamma_k+\Gamma_k^\top
]

保持 reversal-even component。

2024 STVNN 正是从 sample covariance 出发建 multivariate relational model，并专门研究了 streaming/non-stationary 下的稳定性。([arXiv][10])

HLOB 也从另一个方向证明了 LOB 中**变量间非平凡 dependency structure**本身包含可预测信息，而不仅是单通道 trend。([arXiv][11])

---

## 最终网络甚至不需要很复杂

```text
coherence features
+ symmetric lag-cov features
        ↓
     64–256 dims
        ↓
     RealMLP
        ↓
    cosine loss
```

甚至第一轮直接：

```text
Ridge
RealMLP
```

就够判生死。

---

# 八、为什么我没有把下面这些列入 Top 3

### TLOB / LiT

TLOB 同时对 temporal/feature axes 做 attention，并使用 Bilinear Normalization；论文也报告 MLPLOB 这种简单 structured MLP 很强。([arXiv][12])

而且它在 Tesla/Intel 数据里加入 message/order information 后，F1 约增加 1.5 个点。([arXiv][12])

**但我仍然不建议直接跑 TLOB。**

因为它回答的是：

> “如何 encoder 已有 LOB。”

而不是：

> “我们以前没利用哪个随机变量？”

你已经有 Transformer OOD collapse。

所以 TLOB 对我们最大的价值是：

> **BiN / structured MLP / feature-time mixing 的 inductive bias**

不是整套 Transformer。

---

### MLPLOB

这反而很值得借。

MLPLOB 输入 last (T) snapshots，先 feature-mixing MLP，再 temporal-mixing MLP，使用 residual + LayerNorm + GeLU。([arXiv][12])

TLOB/MLPLOB 还显式使用 Bilinear Normalization 来应对 financial non-stationarity。([arXiv][12])

这和 RealMLP 在你这里成功非常一致。

但：

> **MLPLOB 本身不是下一次 0.01。**

它应该成为 Candidate 1/2 的小 encoder，而不是独立主线。

---

### HLOB

它使用 Information Filtering Network/TMFG 去捕捉 volume levels 之间复杂 dependency。([arXiv][11])

非常支持你的：

[
cross-channel>single-channel
]

发现。

但是直接复制 HLOB，又变成：

> 更复杂的 Market encoder。

因此我只迁移它的**higher-order inter-channel dependency 思想**到 RICS。

---

### LOBERT

2025 LOBERT 把完整 multidimensional LOB message 当 token，同时保留 continuous price/volume/time，并取得很强 message/midprice prediction。([arXiv][13])

但你 TinyLOBERT latent 已经和 152 features：

[
corr=0.86\sim0.98
]

所以这条路线**继续 KILL**。

---

### LOBDIF

LOBDIF 用 diffusion 学 joint event-time distribution，并且有官方代码。([arXiv][14])

理论上很漂亮。

但 MSCapital 当前：

* long sequence dependency 弱；
* OOD 极危险；
* 我们不是生成整个 LOB。

所以现在上 diffusion 是严重 overkill。

我只借它一个思想：

[
P(O|M),\quad P(T|M,O)
]

而不是复制 diffusion。

---

# 九、最值得先跑的三个实验

## Experiment 1 — MAG-MoE Probe

### Hypothesis

Market 的强 magnitude signal 可以识别：

> **什么时候现有 event alpha 值得信。**

### Implementation

完全不需要重训大模型。

已有：

```text
v7 OOF prediction
market magnitude OOF
```

把 magnitude 转：

```text
rank_mag
P90
P95
```

做 3–4 个 bins。

每个 bin 单独拟合：

[
\hat y=a_k\cdot v7
]

更进一步：

```text
tiny gate
inputs:
  market magnitude rank
  market extreme probability
  liquidity state

experts:
  frozen v7
  frozen cosine-152
  frozen market alpha / other orthogonal predictions
```

只训练 gate。

### Baseline

当前 frozen v7 / 0.142 ensemble。

### Metric

必须同时看：

* frozen 51–70 cosine；
* 61–70；
* monthly Δ；
* extreme deciles；
* test prediction norm；
* corr(new,v7)。

### Continue

我会要求至少：

[
\Delta cosine\ge+0.0007
]

并且：

* ≥14/20 months positive；
* late months 不消失；
* gate 明显不是常数。

### Kill

如果：

[
\Delta<+0.0004
]

或者 gain 全来自 1–2 个 extreme months：

# KILL。

不继续搞深 MoE。

---

# Experiment 2 — SCFI Probe

### Hypothesis

Order/Transaction raw features 中真正有价值的不是绝对流量，而是：

[
\boxed{
flow-surprise|market-state
}
]

### Implementation

不要先写 Hawkes。

先做最廉价版本。

构造约：

[
32\sim64
]

个 Order aggregates。

训练：

[
M\rightarrow OFeatures
]

模型用：

* Ridge；
* small RealMLP；
* 或 low-depth LightGBM。

必须 cross-fit。

得到：

[
Z_O
]

同理：

[
(M,O)\rightarrow T
]

得到：

[
Z_T.
]

只使用：

[
[Z_O,Z_T]
]

训练一个 RealMLP/cosine model。

---

### 最关键的对照组

同维度：

[
[O,T]
]

vs

[
[Z_O,Z_T].
]

如果 innovation 理论对：

应该看到：

[
corr(pred_{innovation},v7)
<
corr(pred_{raw},v7)
]

同时 blend gain 更高。

---

### Continue

要求至少满足两个条件：

[
corr(new,v7)<0.80
]

以及：

[
\Delta blend\ge0.0007.
]

如果能到：

[
+0.0015
]

直接升级 priority #1。

---

### Kill

如果：

[
corr(pred_{raw},pred_{innovation})>0.95
]

且：

[
\Delta blend<0.0004
]

直接证明：

> conditional residualization 没有释放新的信息。

**彻底关闭 Conditional Innovation 主线。**

这点必须写死，不能因为理论漂亮一直做。

---

# Experiment 3 — RICS Probe

### Hypothesis

Market signal 本质是：

[
\boxed{
local\ cross-channel\ phase\ relationship
}
]

而不是传统 temporal forecasting。

### Implementation

最后：

[
10\ timesteps\approx30s
]

计算：

* zero-lag covariance；
* lag ±1/±2/±3 symmetric covariance；
* cross-channel spectral coherence；
* phase difference cosine；
* local energy ratio。

最后压到：

[
64\sim256
]

维。

RealMLP + cosine。

---

### 必须做四臂 ablation

```text
A raw last-10 flatten
B mean/std/trend
C covariance only
D covariance + coherence
```

再跑：

```text
D(original)
vs
D(reverse)
```

理论上应该：

[
pred_D(X)\approx pred_D(reverse(X)).
]

---

### Continue

Market-only：

当前大概：

[
corr(pred,y)\approx0.086
]

所以我要看到至少：

[
corr\ge0.095
]

更理想：

[

> 0.10.
> ]

同时 blend：

[
\Delta\ge0.0006.
]

---

### Kill

如果：

[
corr<0.09
]

或者：

[
corr(RICS,currentMarket)>0.9
]

且无 blend gain：

# KILL。

不要继续做 wavelet / image / fancy spectral CNN。

---

# 十、真正的实验优先顺序

综合**成本 × 信息增量 × OOD × 成功概率**：

[
\boxed{
MAG\text{-}MoE\ Probe
}
]

↓

[
\boxed{
SCFI\ tabular\ innovation
}
]

↓

[
\boxed{
RICS
}
]

如果 Experiment 2 过线：

立刻改变路线：

```text
SCFI-tabular
      ↓
conditional count/intensity
      ↓
marked point-process innovation
      ↓
Order innovation
      ↓
Trade innovation
      ↓
state-conditioned final head
```

而不是继续调 MAG-MoE。

---

# Final verdict

这次调研以后，我会**部分修正**原来“Conditional Innovation 应该直接成为绝对主线”的判断。

## 最可能立刻赚钱的是：

[
\boxed{\text{MAG-MoE}}
]

因为它几乎完全由你已经测出来的事实推出：

> Market 最强的是 magnitude；v7 恰恰在大 magnitude 上方向最准；metric 又恰恰最重视这些样本。

所以它是目前**最高期望值的 +0.001～+0.003 路线**。

---

## 最值得赌真正 breakthrough 的仍然是：

[
\boxed{\text{SCFI}}
]

因为这是三个方案中唯一一个真正尝试提取：

[
I(Y;O|M)
]

和：

[
I(Y;T|M,O)
]

而且 2022 state-dependent neural Hawkes、2024 OFI forecasting、尤其 2026 state-first L2/order-flow 结果形成了一条相当完整的外部证据链。([ACM Digital Library][15])

**如果 MSCapital 还有一个从 0.142 跳向 0.15 附近的隐藏台阶，我目前最愿意押它来自这里，而不是另一个更强的 Transformer。**

---

## 最有可能成为低成本正交 alpha 的是：

[
\boxed{\text{RICS}}
]

而且这一次我不会建议“试试 scattering”。

你的实验已经给出了更强的 inductive bias：

[
\boxed{
short\ range
+
cross\ channel
+
phase\ sensitive
+
reversal\ symmetric
}
]

这实际上指向的不是传统 forecasting network，而是：

[
\boxed{
local\ synchronization\ geometry
}
]

这是这轮调研里我认为最值得新增的一条非主流线。

**因此下一步已经不需要继续查模型名字了。应该让 Hermes/Codex 直接实现这三个 probe，先跑 MAG-MoE，再 SCFI，再 RICS，并严格按上面的 kill criterion 判生死。**

[1]: https://php.lobsterdata.com/info/HowDoesItWork.php "https://php.lobsterdata.com/info/HowDoesItWork.php"
[2]: https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/overview "https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/overview"
[3]: https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970 "https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970"
[4]: https://arxiv.org/abs/2607.09230 "https://arxiv.org/abs/2607.09230"
[5]: https://github.com/ashimoo/State-Dependent-Parallel-Neural-Hawkes-Process-for-LOB-Event-Prediction "https://github.com/ashimoo/State-Dependent-Parallel-Neural-Hawkes-Process-for-LOB-Event-Prediction"
[6]: https://github.com/ashimoo/State-Dependent-Parallel-Neural-Hawkes-Process-for-LOB-Event-Prediction/blob/master/run_model_universal.py "https://github.com/ashimoo/State-Dependent-Parallel-Neural-Hawkes-Process-for-LOB-Event-Prediction/blob/master/run_model_universal.py"
[7]: https://github.com/ashimoo/State-Dependent-Parallel-Neural-Hawkes-Process-for-LOB-Event-Prediction/blob/master/model/model_CT4LSTM_PPP.py "https://github.com/ashimoo/State-Dependent-Parallel-Neural-Hawkes-Process-for-LOB-Event-Prediction/blob/master/model/model_CT4LSTM_PPP.py"
[8]: https://arxiv.org/abs/2408.03594 "https://arxiv.org/abs/2408.03594"
[9]: https://arxiv.org/abs/1807.08869 "https://arxiv.org/abs/1807.08869"
[10]: https://arxiv.org/abs/2409.10068 "https://arxiv.org/abs/2409.10068"
[11]: https://arxiv.org/abs/2405.18938 "https://arxiv.org/abs/2405.18938"
[12]: https://arxiv.org/html/2502.15757v2 "https://arxiv.org/html/2502.15757v2"
[13]: https://arxiv.org/abs/2511.12563 "https://arxiv.org/abs/2511.12563"
[14]: https://arxiv.org/abs/2412.09631 "https://arxiv.org/abs/2412.09631"
[15]: https://dl.acm.org/doi/10.1145/3534678.3539462 "https://dl.acm.org/doi/10.1145/3534678.3539462"
