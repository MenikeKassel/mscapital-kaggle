# RESULTS.md — 实验记录表

> 纪律: One-Question-One-Experiment; 每行记录 = 一个实验。
> 双轨指标: Research (CV cos + 说明切分) / Competition (Public LB)。
> CV 协议 v1: 官方基线协议 = train month 0-50, valid 51-70 (冻结, 除非 validation_protocol_v2 说明理由)。

## B 级基线阶梯

| ID | 模型 | 特征数 | CV cos | CV切分 | Public LB | 备注 |
|---|---|---|---|---|---|---|
| B0 | 全0/常数 | - | - | - | 0 / -0.007(全-1) | sanity check (社区数据) |
| B1 | LightGBM (官方参数) | 90 | **0.130204** | m0-50 / m51-70 | 0.117 (官方) | ✅ 本机复刻完成 2026-08-10, best_iter=700, 总耗时173s |

## 实验日志 (G 级起)

| ID | RQ/Hypothesis | 变更 (Treatment vs Control) | CV cos | LB | 结论 | 日期 |
|---|---|---|---|---|---|---|
| A1 | RQ1: CV对切分敏感性 | 同特征同参数 × 4切分 | CV1 0.1302 / CV2 0.1338 / CV3 0.1352 / CV4 0.1418 | - | 验证期越近CV越乐观; CV1最诚实(与LB 0.117接近); 用CV1作主评估 | 2026-08-10 |
| A2 | RQ2: 特征数量 | 17基础 vs 90完整 (CV1协议) | 0.0969 vs 0.1302 (+0.0333) | - | 增量特征(窗口/EWM/交叉)贡献绝大部分预测力; 复现社区观察(+0.0357), 甜点位真实 | 2026-08-10 |
| B1 | 特征组消融 LGO | 6组逐一移除 (CV1协议) | 窗口-0.0102 / 盘口-0.0042 / 订单-0.0035 / 交叉-0.0003 / 成交-0.0000 / EWM+0.0000 | - | 窗口统计是绝对主力; EWM/成交基础可删(移除无损失) | 2026-08-10 |
| B2 | 特征精简 | 90 vs 84(删EWM) vs 77(+删成交) | 0.130204 / 0.130237 / 0.129744 | - | 删EWM持平(84可采纳); 再删成交略降; 主口径保留90 | 2026-08-10 |
| C1 | 增强窗口特征 | 90 vs 109(+窗口5/30/300+偏度/分位/比率) | 0.130204 / 0.129760 (-0.0004) | - | 特征工程到顶; 90特征即甜点位(复现社区); 转向模型侧 | 2026-08-10 |
| D1 | 超参数单变量 | 官方 vs leaves64/128, lr05, L2_20, minleaf100 | 0.130204 / 0.130763* / 0.130619 / 0.127126 / 0.129931 / 0.128280 | - | *leaves64 小幅提升(+0.0006)采纳; lr0.05/minleaf100 明显伤; 单参数空间有限 | 2026-08-10 |
| E1 | 时间衰减加权 | 等权 vs 线性2x/3x, 指数温和/强 | 0.130204 / 0.130161 / 0.128900 / 0.128297 / 0.125001 | - | 全部无效或有害; 近期加权不能桥接漂移; 旧月份数据量价值更大 | 2026-08-10 |
| F1 | 轻量MLP (90特征) | LGBM vs MLP[256x2] | 0.130204 / 0.132065 (+0.0019) | - | 表格NN温和超越GBDT; NaN需fill(282K NaN@19列); loss未收敛有空间 | 2026-08-10 |
| G1 | LGBM+MLP融合 | w网格 (相关性0.84) | 单0.1308/0.1321 → blend **0.137041** (+0.005) | - | 融合是最大杠杆; 简单平均即优 | 2026-08-10 |
| F2 | MLP 30ep×3seed集成 | 单seed vs 集成 | 0.1321/0.1298/0.1287 → ens **0.133736** (+0.0017) | - | 集成拉回弱seed; MLP-ens更强更稳 | 2026-08-10 |
| G2 | 三模型融合 | +XGBoost | blend (0.1/0.4/0.5) **0.137710** | - | XGB单模型0.1311; 树间相关0.96, MLP互补0.83 | 2026-08-10 |
| G3 | 三模型融合v2 | MLP-ens替换 | **0.138931** (0.1/0.4/0.5) | - | 超过社区CV4最佳0.1389(CV1诚实协议下) | 2026-08-10 |

## 最终提交 (2026-08-10)

- 文件: `output/submissions/submission_blend_v1.csv` (647,896行, 无NaN, sample_id连续)
- 构成: LGBM(全量443iter) + XGB(全量815iter) + MLP(全量30ep×3seed), 权重 0.1/0.4/0.5
- 离线: CV1 blend = **0.138931** (vs B1锚点 0.130204, 提升 +0.0087)
- **Public LB = 0.122 → 排名 #79/100** (2026-08-11 00:22)
- CV-LB 校准: CV1 - LB ≈ 0.017 (分布漂移量级); 超官方基线(0.117) +0.005
- 提升空间: 前10需 0.151 (差距 0.029); 公开最强单模型 RealMLP 0.134

## 第二轮迭代 (2026-08-11 00:47)

| 实验 | 结果 |
|---|---|
| H1 五模型 | CatBoost 单模型 0.1334 (惊喜); MLP2 512宽持平; 4模型融合 **0.139332** |
| v2 提交 (XGB+Cat+MLP 0.1/0.4/0.5) | **LB 0.122** (与v1相同) |

**校准结论: 表格融合路线 LB ≈ 0.122 封顶** (CV 0.1393 → LB 0.122, 漂移差 0.017)。
CV +0.0004 未转化为 LB 提升 → 表格特征+树+MLP 组合已饱和。
突破方向: 序列模型 (Transformer/DeepLOB, Kaggle GPU) + 0726风格高级特征 (大单检测/VWAP/动量)。

## P0 阶段 (2026-08-11, GPT 双拍板执行)

### P0-2 Adversarial Validation ✅ (20_adversarial_validation.py)

| 组 | 对比 | AUC | 解读 |
|---|---|---|---|
| A | m0-50 vs m51-70 | 0.7334 | train 内部漂移已明显 (持续演化, 非 test 独有) |
| B | m51-70 vs test | 0.7772 | 近期 vs test 漂移最大 |
| C | full train vs test | 0.7659 | 整体漂移确认 |

- **预测主力 = 漂移主力**: m_sp_mean (三组TOP1, C组gain是第2名1.4倍), m_depth_mean, m_rv, o_vol_sum/o_n_120
- 特征组漂移: m_book 45.1% > o_order 21.8% > m_window 18.6% > t_transaction 10.6% > x_cross 2.4% > m_ewm 1.5%
- **CV-LB 差 0.017 的机制解释**: 价差/深度/波动特征在 train→test 系统性漂移, 模型学到的是 m51-70 regime alpha
- E1 时间加权无效的解释: 漂移是市场状态演化, 不是简单时间距离
- 方向: 归一化/invariant 表示 (相对价差 spread/mid, 相对深度) 是下一关键实验

### P0-1 Temporal Matrix ✅ (19_temporal_matrix.py)

| fold | lgb | xgb | cat | mlp | blend |
|---|---|---|---|---|---|
| T1 (m31-40) | 0.11826 | 0.11558 | 0.11756 | 0.11062 | 0.12047 |
| T2 (m41-50) | 0.12360 | 0.12055 | 0.12604 | 0.11611 | 0.12726 |
| T3 (m51-60) | 0.12714 | 0.12603 | 0.12760 | 0.11961 | 0.13000 |
| T4 (m61-70) | 0.13395 | 0.13539 | 0.13850 | 0.13346 | 0.14517 |
| H1 (m41-50) | 0.11903 | 0.11902 | 0.12172 | 0.11414 | 0.12428 |
| H2 (m51-60) | 0.12743 | 0.12518 | 0.12705 | 0.11821 | 0.12883 |
| PSEUDO38 | 0.12423 | 0.12230 | 0.12482 | 0.11963 | 0.12916 |

- **CV1 模型选择能力证伪**: MLP CV1 第一(0.1337)但 7/7 temporal folds 垫底(mean 0.1187) → MLP 优势是 m61-70 regime 特有
- **CatBoost 最稳健单模型** (mean 0.1264); blend 每 fold 全胜 (+0.003)
- 月度 cos 随月份上升 (61-70 > 51-60) → test 期可预测性可能更高; half-life 不适用 (λ<0)
- 排序 Spearman: T1/T4 与其余 fold 相关 0.70, 其余 0.90+ (大体稳定但有翻转)

### P0-3 权重重估 ✅ (21_weight_reopt.py)

- temporal-mean 最优: (lgb 0.2, xgb 0.0, cat 0.5, mlp 0.3) — **MLP 降权, CatBoost 升权**
- Pseudo-LB38: 旧 0.129155 → 新 0.130015 (+0.0009)
- **v3 提交 (temporal 权重): LB 0.122** — 三次提交 (v1/v2/v3) 全部 0.122 → **表格路线 0.122 平台期确认**, 权重微调无法突破
- P0 决策门: 情况 A+B 混合 (排序大体稳定但 CV1 会翻转) → robust model selection 建立 (temporal mean + Pseudo); 突破需 P1 结构性变化

### P0.5-B MLP Fairness Check ✅ (24_mlp_fairness.py)

| fold | 原矩阵(last-epoch) | 30ep+best-state | 60ep+best-state |
|---|---|---|---|
| T1 | 0.11062 | 0.116825 (+0.0062) | 0.118784 |
| T2 | 0.11611 | 0.120613 (+0.0045) | 0.120653 |
| T3 | 0.11961 | 0.125711 (+0.0061) | 0.125495 |

- **原 Temporal Matrix 对 MLP 不公平**: 用 last-epoch 评估, 树模型无此问题; best-state 后 MLP 显著改善
- 30ep vs 60ep 无差异 → 训练轮数不是瓶颈; "MLP 假冠军"修正为 "评估协议高估其劣势 + MLP 确实弱于 CatBoost (T2差0.0054)"
- 待办: 重跑 Temporal Matrix MLP 列 (best-state)

### P0.5-C Drift Intervention (23_drift_intervention.py, 进行中)

| fold | R0(90) | R1(删top10漂移) | R2(归一化替换) |
|---|---|---|---|
| T3 | 0.127596 | 0.126103 (-0.0015) | 0.128653 (+0.0011) |
| T4 | 0.138502 | 0.139277 (+0.0008) | 0.140964 (+0.0025) |
| H2 | 0.127048 | 0.126934 (-0.0001) | 0.129901 (+0.0029) |
| PSEUDO | 0.12482 | 0.12557 (+0.0008) | 0.12764 (+0.0028) |

**P0.5-C 结论: R2 归一化 4/4 folds 全正 (mean +0.0023), 远期 fold 提升最大 → 归一化延长 alpha 寿命 (漂移干预因果链确认)**

### P0.5-D R2 × 全融合 ✅ (25_r2_blend.py)

| fold | R0 blend | R2 blend | 增量 |
|---|---|---|---|
| T3 | 0.131345 | 0.131304 | 持平 |
| T4 | 0.145879 | 0.147596 | +0.0017 |
| PSEUDO | 0.129861 | 0.131316 | +0.0015 |

### 🏆 v4 提交: LB 0.123 (首次突破!)

- **v4 (R2 归一化 + temporal 权重): LB 0.123** vs v1/v2/v3 全部 0.122
- **P0.5 科研验证成功**: PSEUDO +0.0015 → 真实 LB +0.001; "归一化延长 alpha 寿命"在真实榜单验证
- 完整链条: 对抗验证(漂移诊断) → 归一化干预(P0.5-C) → Pseudo 验证(P0.5-D) → LB 确认(v4)
- 教训: 评估协议公平性(MLP last-epoch bug)和表示归一化都是真实杠杆

## P1 阶段 (2026-08-11, 用户拍板: 冻结提交, P1-1/P1-2 后看效果)

### P1-1a 微观结构特征构建 ✅ (27_build_micro_features.py)

- 22 个无量纲 primitive (order: add/cancel imb, OFI norm, arrival rate, burstiness, 大单; tx: 强度/大单/fast-slow flow; market: microprice gap, 相对价差, L2 深度不平衡)
- train/test 各 119s 构建完成

### P1-1b 双轴筛选 ✅ (28_feature_screening.py)

- Alpha 轴 (CatBoost): T3 +0.0001 / T4 +0.0041 / PSEUDO +0.0023, mean **+0.0022**
- Drift 轴: ΔAUC +0.0168 ⚠️ (部分新特征也携带漂移, 需第二轮归一化)
- 判定: 净效果为正, PSEUDO 裁判提升

### P1-1c 全融合验证 ✅ (29_micro_blend.py)

| fold | R2 blend | R2+micro blend | 增量 |
|---|---|---|---|
| T4 | 0.147596 | 0.152536 | +0.0049 |
| PSEUDO | 0.131316 | **0.134871** | **+0.0036** |

- **迄今最大单步提升** (超 R2 的 +0.0015 两倍); 按校准 LB ≈ PSEUDO-0.008 → 预期 LB ~0.127
- 管线: PSEUDO 0.1292 → 0.1313 → 0.1349

### P1-2 TCN 序列模型 ✅ (kaggle_p12_tcn.py + 33/36)

- 双塔 TCN (FAST 60s×16ch + SLOW 60bar×8ch), sequence-only
- 本地 25ep: PSEUDO cos 0.0627; 增强版 60ep×3seed: PSEUDO 0.0738 / T3 0.0670 / T4 0.0523
- **3-fold 融合验证 (34_fusion_3fold.py): 表格+TCN 增益全正** (T3 +0.0035, T4 +0.0023, PSEUDO +0.0044), corr 0.28-0.40
- 远程 Kaggle 3 次 ERROR 根因: ①polars streaming dtype panic (cast 修复) ②Tesla P100 与预装 torch 不兼容 (v3 降级 torch 2.2.2 待验证)

### 🏆 v5 提交: LB 0.125! (2026-08-11 16:15)

- v5 (R2+22微观特征全量融合): **LB 0.125** (v4=0.123, v1-3=0.122)
- 校准: 表格 PSEUDO 0.1349 → LB 0.125 (差 0.010)
- 提交轨迹: 0.122 → 0.123 → 0.125, 全部离线验证兑现

### v6 融合提交 (37_final_v6.py, 待提交)

- TCN 全量 (60ep×3seed, 0-70月) + test 预测, w_tcn=0.07
- ⚠️ **风险信号: test 上 corr(tab, tcn)=0.03** (PSEUDO 上 0.30-0.40) — TCN 全量在 test 分布外可能退化, v6 预期 LB ≈ v5 或略波动

### 🏆 v7 提交: LB 0.135! (2026-08-12 02:19, RealMLP 复刻融合)

- **v7 (v5融合 + RealMLP复刻 0.8/0.2): LB 0.135** (+0.010 单步!超公开 RealMLP 单模型 0.134)
- v7b (0.75/0.25): 0.134 → **0.20 权重是甜点位**
- 0726 特征复刻 (152特征) + RealMLP_RQ (n_ens=16, CV 0.1439@80万切分) 完整复现
- corr(v5表格, RealMLP) = 0.83 (test 上), 融合增益真实
- 提交轨迹: 0.122 → 0.123 → 0.125 → **0.135**

### 🏆 v8 提交: LB 0.142! 排名 #30 (2026-08-12 05:25)

- **v8b (v7 + lb142ref 0.5/0.5): LB 0.142** — 追平最强公开方案 (yangq369 submit-lb142)
- v8a (0.7/0.3): 0.139; corr(v7, ref)=0.82
- lb142 包 (66.9MB): 作者开源推理包 (MultiStream+RealMLP v9/v10, 5成员ens5⊕v10, 公式 0.6·ens5+0.4·v10)
- 提交轨迹: 0.122 → 0.123 → 0.125 → 0.135 → **0.142**; 排名 #82 → #48 → **#30**
- 关键: 公开方案预测文件直接融合是合法且高效的 (作者明确开源)

## P2 校准接手 (2026-08-12, Codex)

### RealMLP PSEUDO / Regime B 首个定标点 ✅

- Kaggle kernel `msc-realmlp-pseudo` v12 COMPLETE；P100 兼容启动后，仅训练 PSEUDO fold，并从最佳 EMA（epoch 9/10）立即生成预测。
- RealMLP 单模型 PSEUDO（m0-32 / m33-70，672,948 行）= **0.138560**；云端日志中心化口径 0.138542。
- 重放 v5 表格 PSEUDO = **0.134871**；与历史 P1-1c 完全一致。
- 实际 v7 原始融合 `0.8 table + 0.2 RealMLP` = **0.139683**（Δtable=+0.004813）。
- v7 Public LB=0.135 → Regime B gap=**0.004683**，小于 Regime A，但仅 1 个点，不建立通用公式。
- Pearson corr(table, RealMLP): valid **0.8658** / test **0.8263**。
- 尺度门禁：v7 valid std=0.003303 / test std=0.002460，test/valid=**0.7447**。这是明确的幅度迁移，但既有 v7 已在 LB 验证有效；新模型不得忽略该信号。
- 原始尺度 PSEUDO 最优约 `w_RealMLP=0.03`（0.141753），只作 forensic 诊断；不据此提交、不继续 Public LB 权重扫描。
- 产物：`output/rlps_v12/realmlp_pseudo_pred.npz`、`v5_table_pseudo_pred.npz`、`v7_pseudo_diagnostics.json`。

## Protocol-v2 C3 Clean Table (2026-08-13)

- 四折可信 outer: PSEUDO **0.135051** / H2 **0.134216** / T3 **0.135454** / T4 **0.150389**；描述性均值 **0.138777**。
- PSEUDO 相对 legacy v5 `0.134871` 仅 `+0.000180`，prediction corr `0.9922`：历史 Table 结论基本真实。
- H2/T3 在完全相同 m51-60 target 上差 `+0.001238`，增加 m41-50 历史对 expanding refit 有稳定价值。
- LGBM/CatBoost/3-seed MLP 固定 `0.2/0.5/0.3` blend 四个 outer 均优于任一单模。
- Table 与 C2 30-epoch RealMLP outer corr 为 `0.81-0.85`，但 RealMLP std 是 Table 的 `34-42x`；C4 必须在 inner 上比较 raw/std/RMS。
- 详见 `docs/c3-clean-table-results.md`；未创建 Kaggle competition submission。

## Protocol-v2 C4 Clean Baseline v2 Frozen (2026-08-13)

- Inner-only raw/std/RMS × `0.00-1.00` 校准，fold-adaptive outer: **0.142649 / 0.141762 / 0.143515 / 0.156924**。
- 生产规则冻结为 `0.63×RMS(30ep RealMLP) + 0.37×RMS(Clean Table)`。
- 各折严格 inner-only 校准均全正，描述性均值 **0.146212**，相对 RealMLP 平均 **+0.005026**，通过 `+0.0005` 门禁。
- 固定 RMS/0.37 规则用各折 inner scale 做相关压力审计，四折 **0.142358 / 0.141795 / 0.143515 / 0.156924**，平均增益 **+0.004961**；此项不是独立无泄漏 outer 估计。
- 统一生产权重是四个 inner 选择的后验聚合；因 T3/T4 inner m41-50 与 PSEUDO outer 重叠，禁止把统一权重回评 PSEUDO 并声称是无泄漏 outer 分数。
- 数值 scale 不从 outer/test 拟合；canonical rolling OOF m51-70 共 353,246 行，融合分数 **0.149173**。
- 生产 RMS scale: RealMLP `0.01481242584`，Table `0.0003505723161`；m61-70 严格历史 block 的两组件分数为 `0.147951 / 0.150578`，冻结融合为 `0.154614`。
- method/weight/scales/component order/scale source 与最终 prediction schema 全部冻结；后续 Alpha 不得回改。详见 `docs/c4-clean-baseline-results.md`。

## P6R 阶段 (2026-08-14, Retrieval-Conditioned Residual Alpha)

### P6R-00 Retrieval Residual Mean ✅ 完成（门禁未过 → KILL retrieval route）

- 设计: E02 11 context 特征为状态表示，canonical OOF (21-70) 严格时序 KNN（month<query 断言），检索残差均值 r̂，y_hat = RMS(y0)+α·RMS(r̂)，α 仅 tune 月选择。预注册 8 候选 = K{64,128,256,512} × {euclidean,cosine}，零追加。
- **32/32 候选×折 Δc4 > 0（无一转负），最大 PSEUDO Δ = +0.000588 (cosine_K512) = gate +0.0015 的 39%**
- corr(r̂,r) = +0.004~+0.010（残差可预测性真实但极弱）；corr(r̂,y0) ≈ 0（正交 ✓）；normalized MSE 全负（与 E02 同）
- **门禁失败**: 无候选同时满足 bootstrap CI 下界>0 与 α≥0.10（PSEUDO cosine_K512 CI=[−0.00013,+0.00128]；α 几乎全卡网格下界 0.05）
- 邻居质量健康（月份熵 3.24/3.91、平均覆盖 27/50 月）→ 问题不在检索机制，在"相似状态的残差均值本身预测力弱"
- **科学结论**: E02 的窗口内 residual cosine 0.013 不可变现为跨月预测增量 → "残差可解释性 ≠ 残差可预测增量"；与 P4-06A/P5-02I probe/P5-03 负面链第 5 个独立确认
- 推荐: KILL retrieval prediction route；可选终裁 = P6R-01 Local vs Global Ridge（需用户拍板）
- 详见 `docs/p6r_experiment_report.md`

## 负面结果

| ID | 实验 | 结果 | 教训 |
|---|---|---|---|
| N005 | v6 融合提交 (表格+TCN 0.07) | **LB 0.082** (vs v5 0.125) | **序列模型 test 分布外严重退化**: PSEUDO 验证对 TCN 不可靠 (PSEUDO 融合+0.004, 真实 test 灾难); test 上 corr(tab,tcn)=0.03 是致命预警; 表格 PSEUDO 校准可靠 (0.1349→0.125), 序列模型完全不可信 |
| N006 | P1-1e 新特征相对化第二轮 (38_micro_rel2.py) | T4 +0.0003 / PSEUDO **-0.0012** | 到达率/强度类特征相对化在 PSEUDO 反而更差; 第二轮特征工程不采纳; 微观特征第一轮 (P1-1c +0.0036) 已是最优 |
| N001 | 自定义 cos loss (社区 Spiritmilk) | CV↑ LB↓ | 标准loss训练+cos评估; 已吸收进协议 |
| N002 | >90特征 (社区 bestwater) | CV 0.1409 ↑ / LB 0.116 ↓ | 90特征甜点位; 特征增益需LB验证 |
| N003 | Transformer 深度优化 (社区 bestwater) | CV 0.1549 / LB 0.120 | CV/LB脱节=分布漂移, 序列模型不直接加分 |
| N007 | RealMLP PSEUDO v10/v11 运行管线 | v10 在 P100 上重复安装 torch 并无限 `execv`; v11 虽修复启动，但原脚本会先跑无关全量训练，且最终 PSEUDO 预测未加载最佳 EMA | v12 加版本幂等保护，并在 PSEUDO 训练后恢复最佳 EMA、保存产物、立即退出；启动/产物顺序均加入回归测试 |
