# MSCapital 新方法与相似任务调研（2026-08-13）

## 结论

这轮不建议先换成更大的 Transformer/Mamba。当前更值得做的是补齐三类能力：

1. **更强的尺度不变表示**：让同一种微观结构形态跨月份仍可比较。
2. **事件条件强度与局部形态**：提取 152 Dynamics、M01–M06 尚未直接表达的信息。
3. **先诊断、后适应的漂移处理**：只有证明残差依赖市场状态，才允许做局部校准。

建议首批顺序：

```text
E01 ReVol-lite 尺度不变表示
→ E02 Reconditionor-lite 残差—上下文诊断
→ E03 月份/状态稳定相关性门禁
→ E04 Hawkes-surprise 事件条件强度
→ E05 ModernNCA/TabR 因果检索
→ E06 MultiRocket/catch22 局部形态摘要
```

所有方法当前状态均为 **未运行**。论文结果和其他比赛结果只是迁移依据，不是 MSCapital 成绩。

## 项目约束与去重依据

当前自研 Public LB 锚点是 `0.135`。Canonical Clean Baseline 已建立严格滚动 OOF；M01–M05 的现有候选主要表现为小幅正增益，但没有通过 `PSEUDO >= +0.0015` 的晋级门槛。已知失败案例 TCN 在本地看似有效、Public LB 却跌至 `0.082`，说明新方法还必须通过预测尺度、测试分布和相关结构门禁。

因此，这轮排除了以下重复或不合适方向：

- 不重复 M01 的 OFI、窗口事件率、快慢流量差。
- 不重复 M02 的 mid-centered L1/L2 depth geometry。
- 不把 Path Signature、Triplet/Urgency/Depth Pressure、固定欧氏 KNN 换个名字重跑。
- 不恢复 DeepLOB/TLOB/LB142 式大型序列网络。
- 不把外部模型、论文 benchmark 数字或 Kaggle 方案算入自研成绩。

## P0：先跑的低成本实验

### E01 ReVol-lite：窗口级收益—波动—尺度归一化

- **对象类型**：特征/表示。
- **来源任务**：ReVol 面向跨市场股票预测，论文通过收益、波动率、价格尺度归一化处理分布漂移，并在预测时重新引入样本属性。
- **与现有 R2/M02 的差异**：现有 R2 只是八个固定比值，M02 主要对盘口价格和深度做相对化；ReVol-lite 对一个样本窗口内的整组价格变化、深度、成交量和事件强度使用同一局部尺度。
- **MSCapital 最小实验**：
  - 价格变化、spread、microprice gap 除以窗口 realized-vol 或稳健 MAD；
  - 深度、成交量、订单/成交事件数分别除以自身窗口 RMS/MAD；
  - 原始 volatility、MAD、绝对深度和绝对事件率作为旁路保留；
  - learner 固定为现有 residual CatBoost，候选晋级后才复核 RealMLP；
  - 只比较 `raw`、`normalized`、`raw+normalized` 三组。
- **门禁**：同一 Protocol-v2；PSEUDO 增益、4 个外层折稳定性、预测尺度与相关性全部复用现有标准。
- **风险**：原论文主要是日频 OHLC；过度归一化会删除绝对流动性和市场 regime。必须保留 raw 旁路。
- **状态**：未运行。
- **一手来源**：[ReVol 论文](https://arxiv.org/abs/2508.20108)、[作者仓库](https://github.com/snudatalab/ReVol)。

### E02 Reconditionor-lite：先证明残差是否依赖市场状态

- **对象类型**：诊断；本身不产生可提交成绩。
- **来源任务**：Reconditionor 用预测残差与时间上下文之间的依赖检测 context-driven distribution shift；SOLID 再从相似上下文选择局部样本适配预测层。
- **与 M05 KNN 的差异**：M05 直接用固定低维状态和欧氏邻居预测 residual；这里第一步只做诊断，不预设 KNN 有效，也不使用外层目标调参。
- **MSCapital 最小实验**：对 canonical OOF residual，按以下历史可得上下文分桶/建模：
  - volatility、spread、depth、OFI、trade imbalance、event rate；
  - 窗口内 change-point/run-length；
  - 月份位置与近期/长期 regime 距离。
  报告 residual mean、residual RMS、方向偏差、互信息/HSIC 或交叉拟合可预测度。
- **进入 E05 的条件**：至少三个外层折确认同一上下文结构，且交叉拟合 residual 可预测度不是由月份 ID 单独驱动。
- **风险**：外层目标绝不能参与上下文、邻居数、阈值或特征选择。
- **状态**：未运行。
- **一手来源**：[Reconditionor/SOLID 论文](https://arxiv.org/abs/2310.14838)、[作者代码](https://github.com/HALF111/calibration_CDS)。

### E03 Stable-correlation gate：检查增益来自哪些月份和状态

- **对象类型**：验证基础设施，不是模型。
- **动机**：当前许多候选“均值略正但 PSEUDO 不够”，只看全样本 cosine 会掩盖增益集中于少数月份/状态的问题。TabReD 也显示，使用时间切分后模型排名会改变，简单 MLP/GBDT 在真实漂移表格任务上仍然很强。
- **最小实现**：每个候选额外报告：
  - monthly delta mean / median / worst；
  - positive-month ratio；
  - delta 随月份的斜率；
  - 高/低 volatility、spread、depth、event-rate 状态下的 delta；
  - 候选 residual 与 Clean Baseline/RealMLP 的状态条件相关性；
  - 单一月份对总增益的贡献集中度。
- **建议硬规则**：不得用这组细分结果反复选择特征；它用于否决脆弱候选。若要据此生成新候选，必须注册为下一实验。
- **状态**：未运行。
- **一手来源**：[TabReD](https://arxiv.org/abs/2406.19380)、[Group DRO](https://arxiv.org/abs/1911.08731)。

## P1：真正新增信息或模型能力

### E04 Hawkes-surprise：事件条件强度与惊奇度

- **对象类型**：事件表示；首版仍交给 residual CatBoost/RealMLP。
- **核心思想**：M01-A 统计窗口总流量和事件率，但没有表达“在当前历史条件下，这个事件有多意外”以及事件类型之间的激发/抑制关系。Hawkes 模型直接建模不规则事件流的条件强度。
- **事件语法**：先用 6–8 个可稳定重建的 mark，而不是神经 Hawkes：
  - order add/cancel × buy/sell；
  - transaction buy/sell；
  - 可选：是否伴随 L1/L2 quote move。
- **最小特征**：
  - 每类事件的指数衰减强度（fast/medium/slow）；
  - observed count minus expected count 的 standardized surprise；
  - buy→buy、sell→sell、cancel→trade 等少数预注册 cross-excitation；
  - event-time compensator residual、最近一次异常 burst 的距离；
  - 5/15/30/60 秒聚合后保持固定维度。
- **与 M01 的差异**：M01 是无条件窗口统计；E04 是有历史条件的事件到达概率和跨类型激发。
- **最小可行版本**：先做折内估计的指数核/Poisson-Hawkes-lite，不训练 LSTM Neural Hawkes；只在通过后升级。
- **风险**：核参数必须只用历史训练月份估计；事件 mark 定义错误会把同一信息重复多次。参数数量需要强约束。
- **状态**：未运行。
- **一手来源**：[Hawkes-based LOB crypto forecasting](https://arxiv.org/abs/2312.16190)、[Neural Hawkes LOB event model](https://arxiv.org/abs/2502.17417)。

### E05 ModernNCA/TabR：因果、可学习的历史检索

- **对象类型**：learner/检索策略。
- **核心思想**：固定欧氏 KNN 失败不等于“邻居方法无效”。ModernNCA 学习邻居度量；TabR 以注意力式检索将训练样本的特征与标签用于预测。
- **与 M05 的差异**：
  - M05：人工 16 维状态 + 固定欧氏距离 + 历史 residual 平均；
  - E05：在 inner history 学习 embedding/相似度，并显式限制候选记忆库来自 query 之前的月份。
- **最小实验**：
  1. 只在 E02 确认 residual-context 依赖后启动；
  2. 输入先用 32–64 个预注册状态/Dynamics 特征；
  3. 每个 query 的检索库满足 `source_month < query_month`；
  4. 先运行 ModernNCA-lite 或冻结 embedding + ridge weighted-neighbor residual，不直接上完整 TabR；
  5. 比较固定 KNN、learned metric、no-label retrieval 三个消融。
- **风险**：标签泄漏、全量近邻的内存/延迟、query 与 memory 的月份分布差异。所有索引和 scaler 必须折内重建。
- **状态**：未运行。
- **一手来源**：[ModernNCA](https://arxiv.org/abs/2407.03257)、[TabR](https://arxiv.org/abs/2307.14338)。

### E06 MultiRocket/catch22：低成本局部形态表示

- **对象类型**：固定维度序列表示，不是大型序列网络。
- **核心思想**：Path Signature 强调整条路径的几何积分；Rocket 类随机膨胀卷积更擅长发现局部 motif、持续时间与时间位置，catch22 提供自相关、异常事件时点、谱能量、可预测性和非线性动态摘要。
- **最小实验**：
  - 序列只选 6–8 条：mid return、spread、L1/L2 imbalance、OFI、trade imbalance、event rate；
  - 先运行 `catch22-selected`（每条只保留 6–10 个与已有 152 Dynamics 非重复的统计量）；
  - 再对同一序列运行 MiniRocket/MultiRocket transform，使用 ridge 预测 residual；
  - 只有 transform residual 与 Clean Baseline 相关性足够低且通过门禁，才把少量降维特征交给 RealMLP。
- **风险**：catch22 原 benchmark 主要是分类；Rocket 原方法也不是金融收益回归。61 点左右的短序列可能限制长 dilation 的价值。
- **状态**：未运行。
- **一手来源**：[catch22](https://arxiv.org/abs/1901.10200)、[MiniRocket](https://arxiv.org/abs/2012.08791)、[MultiRocket](https://arxiv.org/abs/2102.00457)。

## P2：第二梯队候选

### 冻结式小型 LOB 自编码表示

用历史训练窗口做重构/遮盖恢复，冻结 8–32 维 embedding，再与 152 Dynamics 合并。它符合“表示优先”，但下游论文多为重构、欺骗检测或 mid-price 分类，迁移到 cosine 收益回归仍需验证。先做轻量 encoder，不做生成大模型。

来源：[LOBench](https://arxiv.org/abs/2505.02139)、[相关 LOB 表示学习实现](https://github.com/Financial-Simulation-Lab/LOBench)。状态：未运行。

### RevIN + SAM 小序列消融

不是恢复大 Transformer，而是在同一个浅 MLP/TCN 上做四格对照：none / RevIN / SAM / RevIN+SAM，多 seed 检查稳定性。只有表示先过门禁才考虑 channel-wise attention。

来源：[SAMformer（ICML 2024）](https://proceedings.mlr.press/v235/ilbert24a.html)。状态：未运行。

### Structured LOB image

将事件映射为 `time bin × relative-price bin × {add,cancel,trade} × side`。首版只提二维固定摘要或 tiny-CNN embedding，不训练 diffusion。它比普通多流序列更显式保留“事件发生在盘口哪里”。风险是当前数据只有两档盘口，price-bin 信息可能不足。

状态：未运行。

### DoubleAdapt / month-group DRO

两者都针对漂移，但先后顺序应靠后：DoubleAdapt 依赖可因果更新的标签时序；Group DRO 容易牺牲平均 cosine，且月份组权重很可能过拟合。只在 E03 确认存在稳定 worst-month 模式后，尝试最后一层 adapter 或温和的 worst-month regularizer。

来源：[DoubleAdapt](https://arxiv.org/abs/2306.09862)、[Group DRO](https://arxiv.org/abs/1911.08731)。状态：未运行。

## 相似公开数据集与任务

| 数据集/基准 | 数据与任务 | 与 MSCapital 的相似处 | 适合用途 | 关键限制 |
|---|---|---|---|---|
| [China LOB Benchmark](https://github.com/hkgsas/LOB) | 数千只中国股票，按秒预测未来 1–300 秒 VWAP 变化与成交量 | 多资产、高频盘口、连续目标、多 horizon | 测试尺度归一化、跨资产表示和 horizon 稳定性 | 市场制度和字段与比赛不同 |
| [FI-2010](https://arxiv.org/abs/1705.03233) | 5 只 Nordic 股票、10 档 LOB、约 400 万事件、日序 anchored CV、mid-price 分类 | 不规则事件流、按时间验证 | Rocket/catch22/浅序列 smoke test | 数据老、已预处理、分类目标；不能用它替代项目 OOF |
| [LOBSTER](https://data.lobsterdata.com/info/DataStructure.php) | NASDAQ message + 多档 orderbook，submission/cancel/execution，毫秒至纳秒时间戳 | 原始订单事件与盘口同步，最接近微观结构数据形态 | Hawkes/event grammar 的外部 sanity check | 主要为学术访问，规模大；资产/市场差异明显 |
| [LOBench](https://arxiv.org/abs/2505.02139) | 中国 A 股 LOB 表示的重构、趋势预测、缺失恢复 | 明确把表示与 downstream learner 分离 | 自编码表示与可迁移性 benchmark | 论文任务不是未来收益 cosine |
| [DSLOB](https://arxiv.org/abs/2211.11513) | 合成 LOB，带有标注的压力情景/OOD shift | 能控制分布漂移 | 检验 ReVol/Reconditionor/稳定门禁是否真能识别 shift | 合成结果不等于真实市场 alpha |
| [Hyperliquid Level-4 sample](https://zenodo.org/records/18184441) | BTC/ETH/SOL 原始 book diff、订单状态与交易；论文提供公开样本 | 更细事件语法、公开可复现 | 检验 Hawkes marks、event token 和 rejected-order 思路 | crypto/perpetual 市场；MSCapital 本身没有钱包身份和 rejected order 字段 |

## 相似比赛与可迁移经验

| 比赛 | 任务/指标侧重点 | 可迁移内容 | 不应照搬 |
|---|---|---|---|
| [Optiver Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close) | 盘口与集合竞价的短期价格运动 | imbalance/urgency、横截面 index、近期重训的消融方式 | 零和后处理只有先证明目标结构才可用；集合竞价结构不等于连续市场 |
| [Ubiquant Market Prediction](https://www.kaggle.com/competitions/ubiquant-market-prediction) | 匿名金融表格、未来目标、相关性指标 | 时间切分、相关性稳定性、简单强表格 learner | 随机 KFold 和直接按 investment ID 记忆 |
| [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/overview/abstract/abstract) | 强非平稳、流式推理、weighted zero-mean R² | recent/static、严格流式无前视、漂移诊断 | 目标权重、在线标签和 responder 结构不能假设在本比赛存在 |
| [G-Research Crypto Forecasting](https://www.kaggle.com/competitions/g-research-crypto-forecasting) | 多资产分钟数据、residualized return、相关性评价 | 市场因子残差、时间 API、缺失资产与断点处理 | 分钟 OHLC 特征不能替代订单事件表示 |

## 建议工程排期

### 第一批：同一周内可判生死

1. **E03 稳定相关性报告**：先补验证输出；不训练模型。
2. **E01 ReVol-lite**：先做 20–40 个预注册 normalized/raw-pair 特征；Residual CatBoost 四折。
3. **E02 Reconditionor-lite**：只用 canonical OOF 做残差—上下文诊断。

### 第二批：需要真实新信息

4. **E04 Hawkes-surprise-lite**：指数核、少量事件 mark、固定维度，先不使用神经点过程。
5. **E06 catch22-selected**：比 Rocket 更便宜，先筛是否存在非线性动态信息。
6. **E05 ModernNCA-lite**：只有 E02 通过才启动；严格 past-only memory。

### 第三批：第一、二批至少一项通过后

7. MultiRocket residual ridge。
8. 冻结式小型自编码 embedding。
9. RevIN/SAM 小序列消融。
10. Structured LOB image、DoubleAdapt 或完整 TabR。

## 统一验收规则

- 所有方法默认标记“未运行”，不得展示推测性 MSCapital 分数。
- 所有 scaler、事件强度、embedding、邻居索引和特征选择均折内拟合。
- 外层 target 只用于最终评分，不用于选窗口、mark、邻居数、归一化方式和融合权重。
- 沿用 `PSEUDO delta >= +0.0015`、至少 3/4 外层正增益、worst delta、prediction-scale 和 test-correlation 门禁。
- 新增 E03 的 monthly/state stability 只做否决门禁；若由诊断产生新假设，注册成独立实验再运行。
- 不创建 Kaggle submission；候选过本地协议后再单独决定是否提交。

## 最终推荐

如果只能选一个新实验，选 **E01 ReVol-lite**：成本低、与项目曾经成功的 R2 无量纲化同源，但覆盖范围更系统。

如果只能选一个真正新增的信息源，选 **E04 Hawkes-surprise-lite**：它建模的是事件的条件到达与跨类型激发，不是 M01 已有窗口计数的微调。

如果只能选一个新 learner，选 **E05 ModernNCA-lite**，但前提是 E02 先证明 residual 与市场上下文存在可重复依赖；否则固定 KNN 的失败很可能会重演。
