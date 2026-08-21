# O→T Lag Response 探针方案 v2 (2026-08-15, 吸收 GPT1+GPT2 双评审)

> 状态：**G0 FAIL (2026-08-15) — O→T lag response 主线 STOP**，见 docs/p8-01a-otlag-results.md
> 上游：docs/gpt-review-otlag-2026-08-15.md（v1 方案）→ GPT1/GPT2 双评审 → 本 v2
> 核验：两份评审的全部可实测论断已逐条验证（数学/论文数字/代码事实），**无需纠正**

---

## 1. 双评审共识（两份都同意，直接采纳）

| # | 共识 | 执行 |
|---|---|---|
| 1 | 研究方向 GREEN：O→T 是 method-space 空洞（152 无跨表时序，代码级证实） | 立项成立 |
| 2 | 原 corr(feature, y) 主门禁 RED | 降级为 diagnostics |
| 3 | corr<0.7 正交门禁 RED | 删除（pairwise corr ≠ 正交性） |
| 4 | 必须加 forward/backward placebo（时间箭头） | 新增 |
| 5 | 共同活动度混淆（shared activity）是最大风险 | 新增 matched-activity null + 控制变量 |
| 6 | 最终门禁 = 对现有 baseline 的增量 Δcosine（OOF），不是 raw corr | 重写门禁 |
| 7 | Stage 3 序列模型暂缓（TCN 前科 + CrossStream 是"兑现模型"不是"寻找模型"） | 推迟 |
| 8 | lb142 归因降级（CrossStreamMixer = supporting evidence，不能证明 O→T） | 改表述 |
| 9 | 更便宜路径：无 ML 的 Stage 0 诊断先做 | 新增 |
| 10 | 右删失（censoring）处理必须新增 | 新增 |

## 2. 双评审分歧（合并方式）

| 分歧点 | GPT1 | GPT2 | 合并 |
|---|---|---|---|
| 统计显著性 | block bootstrap 95% CI + permutation null（多 lag 时对 max\|stat\|） | forward>backward + 14/20 月 Δcos | 主门禁用 bootstrap CI；月度正比例降为 descriptive |
| Stage 0 对照 | 三控：within-day shift / negative lag / marginal shuffle | 核心 A(k)=C(+k)−C(−k) asymmetry | 全做：asymmetry 是主统计量，三控是 placebo 组 |
| Stage 1 特征数 | 固定 3–5 个 summary，禁止 HPO | 20–40 个，四组设计 | 20–40 个四组（B 组 cancel→trade 是亮点） |
| α 搜索 | 未提 | α* 解析解 (bA−aC)/(aB−bC) | 采纳解析解（少一个研究自由度） |
| Stage 2 | 同时测表格 baseline + RealMLP（flat ≠ 无用） | 同 | 采纳 |
| Stage 3 桥梁 | learnable response kernel（30 权重或 4 basis） | 推迟到信息兑现后 | 采纳（先 kernel 后 CrossStream） |

## 3. 最终方案：P8-01A → P8-01B → P8-01C → P8-02

### P8-01A：时序箭头诊断（无 ML，半天，最便宜）
- 0.5–1s bin 事件流：O(t)=signed order flow（buy_new/sell_new/buy_cancel/sell_cancel），T(t)=signed tx flow
- C(k)=Σ_t O(t)T(t+k)，k ∈ {−30,−10,−5,−3,−1,0,+1,+3,+5,+10,+30}
- **主统计量 A(k)=C(+k)−C(−k)**（forward/backward asymmetry；A(k)≈0 ⇒ activity cluster 而非 response）
- Placebo 组（三控）：① within-day block shift ② T→O 反向 ③ marginal-preserving shuffle；+ matched-activity null
- 右删失：方法 A（eligible events：k=10 只用 t≤50s 的 order；k=30 只用 t≤30s）
- 实现坑（GPT1 六条 + GPT2 补充，全部写进代码注释）：同 timestamp（算 lag=0）、午休/开收盘边界、日内 U-shape 归一化、规模暴露（volume normalize）、order/tx 机械重合确认、**feature cutoff P0 leakage 单元测试**
- 输出：response curve + asymmetry + placebo 对比
- **Gate G0**：forward response 显著超过 shift/matched null（block bootstrap 95% CI 下界 >0）→ 否则 STOP O→T

### P8-01B：增量探针（若 G0 过，一天）
- 20–40 个特征，四组（禁止扩大搜索）：
  - A. Order→trade conversion：same/opposite side response + ratio（1/3/5/10s）
  - B. **Cancel→trade**：ask cancel→future buy tx、bid cancel→future sell tx（liquidity withdrawal→aggressive execution）
  - C. Response latency：time-to-first-same-side-tx 的 median/p25/p75、fraction<1s/<3s
  - D. Forward-backward asymmetry：future_k − past_k（天然消 activity）
- 全部 OOF：Ridge/ElasticNet/极小 LGBM → q_ot
- **Gate G1**：OOS effect 95% CI 下界 >0（block bootstrap；多 lag 搜索时对 max|stat| 建 permutation null）
- 指标三口径同录（raw Pearson / centered / uncentered cosine），门禁以 **OOS uncentered Δcosine** 为准
- 控制变量（Gate C）：order/tx count、total volume、volatility、spread、152 baseline prediction 后仍有增量

### P8-01C：+152/RealMLP 冻结测试（若 G1 过）
- 固定：fold/seed/model/training budget 全同，只加 O→T 特征
- 双模型同测：表格 baseline(152) ± O→T；RealMLP(152) ± O→T（**RealMLP flat ≠ O→T 无用**：可能是 heavy-tail/threshold 特征与 MLP 交互问题，本身是研究结果）
- α 融合：cos(p+αq, y)，α*=(bA−aC)/(aB−bC)，calibration 区定 α，untouched 20 月验证
- **Gate G2**：paired OOF Δcos 稳定正（≥14/20 月，descriptive）+ bootstrap CI 下界 >0

### P8-02：桥梁模型与 ETCI（仅当 G2 过）
- 先 learnable response kernel：R=Σ_τ w_τ·O_t·T_{t+τ}（4 basis：1–3/3–5/5–10/10–30s）——直接回答"模型喜欢哪个 lag"，比 CrossStream 便宜一个数量级
- 之后才考虑 CrossStreamMixer（lb142 v9 结构借鉴）；预注册 test 侧 corr 结构门禁（v6 TCN 前科）
- ETCI 作为 O→T 第二种表示：hazard = observed post-order hazard / background tx hazard（**不是 raw latency**）

## 4. 版本控制与表述修正（GPT 两处提醒）

1. **lb142 归因降级**：只说"lb142 与现有预测存在显著非共线部分（prediction corr≈0.82，1−ρ²≈33% 残差方差），来源尚未识别；O→T 是待验证假设之一"。
2. **RealMLP 三分**：论文分析 = arXiv 2407.04491v3（2025-01-15 最终修订）；论文代码复现 = standalone/pinned commit（PyTabKit 现版已含 n_ens 等后续机制，不复用为论文证据）；竞赛自定义 = lb142 v10 implementation。
3. **"大样本回归是主战场"降级**：论文 benchmark 1K–500K 且 electricity 类高频任务 MLP 吃亏 → 表述改为"【机制解释】中大型表格回归上 RealMLP 有竞争力；MSCapital 是否在强区仍需本地验证"。

## 5. 待用户拍板（3 项）

1. **P8-01A 的 bin 粒度**：0.5s vs 1s（GPT2 建议 0.5–1s bin，避免毫秒级 event matching 无经济含义）——默认 1s bin + 同 timestamp 单列 lag=0
2. **P8-01B 的 B 组（cancel→trade）**是否保留为第一轮四组之一（GPT2 认为可能比 new→trade 更强；成本相同）——默认保留
3. **资源分配**：P8-01A（半天）先跑，RealMLP 组件迁移（E1-E10）是否并行还是等 O→T 判死后再开
