# MSCapital Plan v1.5.0 — GPT 二轮评审吸收 (Information Audit 前置)

> 日期: 2026-08-14 | 版本: v1.5.0 (上一版 v1.4.0)
> 来源: 用户转述 GPT 二轮评审意见 (P5-02I Information Audit / surrogate data tests)
> 状态: **执行中** — 2026-08-14 用户五项全部拍板通过, 已授权执行 P5-02I (M0-M5 + probes)

## 版本历史
- v1.5.0 (2026-08-14): 吸收 GPT 二轮意见 — 信息论框架确认; P5-02I Information Audit (surrogate 测试) 插入 P5-02 之前; 修正 2 处技术缺陷 (M6 event-time 在 P5-01 协议上不可行、M7 需双源模型); arXiv 论文核验 2 篇 ✅; target 分解 probe + gated alpha 采纳为诊断/可选形态。
- v1.4.0 (2026-08-14): 吸收 GPT 首轮评审 — Alpha Factory 框架; 三处修正 (feature bank 降级、残差聚合版已死、context×event 采纳); 合并 P5 队列。

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
| **S** | **P5-02 序列 latent** (主线, v1.4.0 继承) | frozen market encoder → 32d latent → 拼 152 → RealMLP | corr(latent,152)<0.70 且 ΔPSEUDO≥+0.0015 | **backbone 按 P5-02I 结果选择**: shuffle/reverse 崩 → Conv/SSM; desync 崩 → channel-mixing/attention; block50 崩 → 长感受野 |
| **S** | **P5-02B context×event** (含 M7) | 600s context × 60s event 相对/交互特征 → RealMLP/cosine; M7 错位对照 (对齐 vs 错配) | corr 双轴诊断先行; 展开后 ΔPSEUDO≥+0.0015 且 ≥3/4 折正; M7: 对齐 Δ 显著 > 错配 Δ | 全部相对形式 (防漂移); M7 结果 = 跨源 synergy 直接证据 |
| **S** | **P5-03 序列版残差** | P5-01 cosine 序列模型 → 目标 r = y − β·v7 (M01-A 协议) | corr(res_pred,y)>0 且 corr(res_pred,v7)≈0 且 frozen Δ>0 | 聚合版已死 (P4-06A), 序列版是新审判 |
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
