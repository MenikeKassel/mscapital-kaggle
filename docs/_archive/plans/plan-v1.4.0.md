# MSCapital Plan v1.4.0 — GPT 外部评审吸收 (Alpha Factory 路线)

> 日期: 2026-08-14 | 版本: v1.4.0 (上一版 v1.3.0)
> 来源: 用户转述 GPT 评审意见 (数据生成结构调研之后)
> 状态: **讨论定稿阶段, 未获执行许可** — 等用户明确"开始"

## 版本历史
- v1.4.0 (2026-08-14): 吸收 GPT 评审 — 确立"微观结构 Alpha Mining + 多尺度表示 + 指标对齐 + 正交融合"框架; 修正 3 处与本地证据冲突的论断; 合并 P5 队列。

## 1. GPT 评审核心论点 (与本地证据核实)

GPT 框架: 比赛本质 = LOB/Order Flow Alpha Mining, 不是单模型冠军。双轨 Alpha:
- Alpha A: order+transaction → 152 特征 → RealMLP (MSE/混合)
- Alpha B: market 600s → 新表示 → cosine/corr
- 融合: OOF/residual → cosine-optimal blend

**逐条核实表** (对照本地台账):

| GPT 论断 | 本地证据 | 判定 |
|---|---|---|
| 官方 90 特征 ≈0.125 平台 | v1-v3 三次 LB 0.122 | ✅ 一致 |
| 152 特征+RealMLP 明显突破 | 公开 0.134 → v7=0.135 | ✅ 一致 (严格归因: 特征×架构组合, 4.9 Attribution — 152×树=0.127 反而低于官方特征×树=0.130) |
| TCN/Transformer 没突破 | TCN 真实 LB 0.082 灾难; Transformer CV 虚高 (0.1549/0.120) | ✅ 一致 |
| 152+cosine 仅 +0.001 | 4.14 收口: 两次 Public 0.135, 严格 v7 复刻 +0.001 | ✅ 一致 |
| market-only+MSE 几乎无信号 | P5-01: corr(market,y) = −0.0013 | ✅ 一致 |
| market-only+cosine: corr(y)≈0.086, Δ≈+0.0009, 17/20 月正 | P5-01: 完全吻合 | ✅ 一致 |
| **corr(market, v7)≈0.49 是当前最重要结果** | P5-01: 0.492 → v7 未达信息上限, market 有第二套正交信息 | ✅ 一致 (采纳为路线锚点) |
| MLPLOB 第一梯队 / TLOB-lite 第二梯队 / Transformer 禁起步 | 本地: RealMLP 成功、TCN 失败、TinyLOBERT corr 0.86-0.98 终止 | ✅ 一致 (与 P5-02 backbone 选择同向) |
| objective 只研究 cosine / Pearson / cosine+小MSE | 4.14 收口: cosine 真实但小; 不再展开变体 | ✅ 一致 |
| 停止清单 (TCN 调参/Transformer/152 cosine 变体/RealMLP 搜索/换皮/calibration/套娃) | 与 4.13-4.15 收口一致 | ✅ 采纳 |
| 评估口径升级: Δ+corr(new,v7)+corr(new,y)+月命中率+regime 稳定+frozen OOS | 与 4.5 晋级规则 v2 / 4.11 门禁 / 4.14 三要素一致 | ✅ 采纳 (统一为协议) |

## 2. 三处修正 (GPT 论断与本地证据冲突)

### ⚠️ 修正 1: P5-02A "Market Feature Bank" (手工特征银行) 不能当 S 级主线
GPT 把"600s 手工统计特征家族"排为第一优先。本地证据链已三次证伪聚合/手工形式:
- 34 个聚合版 market 历史特征 (4段×统计): **PSEUDO −0.0002** (4.12)
- P4-01a: 演化 ratio 类特征 (vol/spread/depth 前后段比、mid_trend) 区分力 ≈ 0 — 信息在**水平/密度/波动幅度**, 不在路径形状
- P4-06A: 28 个 4段×7 market-state 聚合特征走残差协议: long 段 PSEUDO **+0.0000** (链条断裂)
- P5-01: 同一 market 数据, **聚合形式无信号, 序列形式 + cosine 有信号** (corr 0.086)

**结论**: market 的信息以**序列/动态形式存在, 手工聚合即丢信息**。GPT 的 feature bank 若跑, 只能是支线 (见 P5-02A-lite), 主线必须是序列表示 (P5-02)。

### ⚠️ 修正 2: P5-03 "market → v7 residual" 聚合版已判死刑, 新实验 = 序列版残差
GPT 提议"预测 v7 的错误 r = y − β·pred_v7"作为新方向。**P4-06A 已用聚合特征测过**: 600s 长上下文解释 LB142-v7 分歧 (R²=0.00117, 18.4× null) 但**不解释 y−v7 残差** (long PSEUDO +0.0000)。方法学铁律: "外部模型用了 X" 的取证发现, 必须测 X→y−own_residual, 两链都成立才值得工程化。
**未测的版本**: P5-01 的 cosine 序列模型 × 残差目标 (聚合版失败 ≠ 序列版失败, 同修正 1 逻辑)。P5-03 重新定义为**序列版残差审判**。

### ⚠️ 修正 3: P5-02B (context × event) 是真正未测的新方向, 采纳为 S 级, 但带预注册门禁
600s context × 60s event 交互 (如 recent imbalance − 600s mean imbalance, recent spread / 600s median) — 台账中确实无此实验, 是 GPT 方案最大增量潜力点。但:
- 历史证据偏弱: 上下文聚合特征 (34 个) 无信号 → 交互特征的增益上限可能也小
- 纪律: 一律**相对形式** (归一化防漂移, 4.7 双轴筛选: Alpha 轴 Δcos + Drift 轴 ΔAUC)
- 预注册门禁: 先做小规模特征家族与 152 特征的 corr 双轴诊断, 再决定展开

## 3. 修正后执行队列 (合并本地 P5 协议 + GPT)

| 优先级 | 实验 | 内容 | 门禁 | 备注 |
|---|---|---|---|---|
| **S** | **P5-02 序列 latent** (主线, 原计划) | frozen market encoder (200步×13ch, cosine, 小Conv1D 禁Transformer) → 32d latent → 拼152进RealMLP | corr(latent,152)<0.70 且 ΔPSEUDO≥+0.0015 | 即 GPT 的 "learned embeddings" + MLPLOB 合体; 冻结防同化 |
| **S** | **P5-02B context×event** (GPT 新方向) | 600s context 统计 × 60s event 相对/交互特征 → RealMLP/cosine | 先 corr 双轴诊断, 展开后 ΔPSEUDO≥+0.0015 且 ≥3/4 折正 | 全部相对形式 |
| **S** | **P5-03 序列版残差** (修正2) | P5-01 cosine 序列模型 → 目标 r=y−β·v7 (M01-A 协议, inner-train 拟合) | corr(residual_pred,y)>0 且 corr(residual_pred,v7)≈0 且 frozen Δ>0 | 聚合版已死, 序列版是新审判 |
| A | P5-04 MLPLOB | 简单 MLP 吃 market 序列表示 (P5-02 backbone 变体) | 与 Conv1D 对比, 同门禁 | P5-02 确认表示有效后才跑 |
| A | P5-05 冻结融合 | v7 + market-feature + market-residual 三 alpha | **41-50 选权重, 51-70 完全冻结** | 4.14 纪律; 提交前 30 秒审计 |
| B | P5-02A-lite geometry 手工特征 | microprice/relative spread/L1-L2 imb/depth slope (未单独测过的家族) | 小规模, corr 诊断先行 | 仅在序列 latent 门禁失败时升级 |
| B | objective 三选一 | cosine / Pearson / cosine+小MSE | 不展开变体 | 4.14 收口 |

## 4. 采纳的评估口径 (统一协议)

实验汇报一律报六维: ΔPSEUDO / corr(new, v7) / corr(new, y) / 月命中率 / regime 稳定 (fold 矩阵) / frozen OOS Δ。单看 Δ 分数不再被接受 (GPT 与本地 4.5/4.11 双重确认)。

## 5. 停止清单 (正式冻结)

TCN 大规模调参 / vanilla Transformer / 152 特征 cosine 花式变体 / 单纯 RealMLP 参数搜索 / CatBoost/XGB/MLP 换皮 / calibration 小修小补 / 同一批特征 ensemble 套娃 / 34 聚合 market 特征重做 / 月漂移后处理 (P4-05 死)。

## 6. 待用户拍板
1. 三修正点是否接受
2. P5-02 / P5-02B / P5-03 的启动顺序 (建议: P5-02 先行, 因 P5-03 依赖其 encoder 基建; P5-02B 可并行)
3. 是否按队列开始执行
