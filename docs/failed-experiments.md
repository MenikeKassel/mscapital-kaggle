# 失败与不足结果库

失败不等于删除；每条记录都保留原因和 do-not-repeat。

## model_saturation

| ID | 路线 | 证据 | 结论 | 原因 |
|---|---|---|---|---|
| C1-FE | R01-table-baseline | negative | 不再盲目堆窗口统计特征 | 特征工程到顶, 90 特征即甜点位 |
| H1 | R01-table-baseline | negative | 不再在表格特征+树+MLP 组合上做小增量融合 | CV +0.0004 未转化为 LB 提升 → 表格特征+树+MLP 组合已饱和 (0.122 平台期) |
| P0-03 | R02-r2-drift | negative | 不再做表格模型权重微调 | 三次提交 v1/v2/v3 全 0.122 → 表格路线平台期确认, 权重微调无法突破 |
| S-06 | R20-submissions | negative | 序列模型不通过 test 侧 corr 结构验证不得进生产融合 | N005: TCN test 分布外严重退化 (PSEUDO +0.004 → LB 0.082), test corr(tab,tcn)=0.03 致命预警 |
| P1-04 | R03-micro-primitives | negative | 不再对微观特征做第二轮相对化 | 相对化在 PSEUDO 反而更差; 第一轮微观特征已是最优 |
| P1-05 | R05-sequence | negative | 不再用序列模型做生产融合 (除非 test 侧 corr 结构验证) | N005: test 分布外严重退化 — PSEUDO 验证对 TCN 不可靠 (test corr(tab,tcn)=0.03 是致命预警); 序列模型完全不可信 |
| P3-02 | R08-unsupervised-latent | negative | 不再用简单拼接做条件化 (FiLM 式融合留给完整版) | 四折全正但拼接稀释 E01 纯状态信号, 未过 gate |
| P3-03 | R08-unsupervised-latent | negative | 不再做掩码预训练 latent | 掩码 latent 无独立信息 (corr 过高), 预注册门禁终止 |
| P3-04 | R08-unsupervised-latent | negative | 不再做网格投影 | 投影无信号 (信息已在聚合特征中) |
| P3-05 | R08-unsupervised-latent | negative | 不投入完整 NHP 模型 | 诊断通过但信号极弱, 不投入完整 NHP |
| P4-08 | R09-hidden-information | negative | 不做残差均值直接预测 | 聚合残差无预测增量 (负面链 #1) |
| P4-15 | R09-hidden-information | negative | 不再追资产身份 | H1+H2 无证据 (与 P4-04b 0.5 价簇区分) |
| M-01 | R06-m-residual | insufficient | 事件流特征无残差增量 | 事件流特征无残差增量 |
| M-02 | R12-geometry-signature | insufficient | 几何特征无残差增量 | 几何特征无残差增量 |
| M-03 | R12-geometry-signature | insufficient | 几何特征线整体关闭 | 几何 temporal 变体无残差增量 (与 M-02 同判) |
| M-04 | R12-geometry-signature | insufficient | 签名特征无残差增量 | 签名特征无残差增量 |
| M-05 | R06-m-residual | insufficient | 交互特征无残差增量 | 交互特征无残差增量 |
| M-06 | R06-m-residual | insufficient | 状态检索无残差增量 (P6R 前身) | 状态检索无残差增量 (P6R 前身) |
| P5-03 | R11-scfi-z | negative | 不再做简单 volatility/幅度 gate; 不再细扫 α∈[0,1] | 嵌套后 gate≈常数 (std=0.011), 月度 6/20; 置换对照≈0; 幅度可预测 ≠ 加权有效 |
| P5-05 | R12-geometry-signature | negative | 不再做 wavelet/shapelet/spectral CNN 短窗形态 | 短窗形态无信息; 相位破坏反演不变; M0-ref 复现 0.0861 确认协议有效 |
| P6R-00 | R13-p6r-production | negative | 不再做检索-残差预测路线 | 无候选同时满足 CI 下界>0 与 α≥0.10; 相似状态残差均值预测力弱; 负面链 #5 |
| P7-01 | R10-amplitude-gate | negative | 不再做 volatility confidence calibration / Market→Confidence 假设 | α 曲线单调下降; 高波动样本方向质量差 (cos 0.215→0.11); baseline 幅度分布已隐式最优 |
| P8-01 | R14-o-to-t | negative | O→T 跨秒响应不存在; 同期lag=0是机械重合(0.33) | 不再做跨表时序箭头探针 (1s bin 尺度判死) |
| P9-01 | R15-p9-quant | negative | 机制性证伪: err_corr≈0.99 — 残差方向无信息(P5-03 AUC 0.51 已证), diversity 无米下锅 | 不再做误差互补类训练 (NCL/反相关损失) |
| C-08 | R17-realmlp-recipe | negative | cosine decay (1→0) 在 30ep 短训练下为负 -0.0009: 后期 LR 过低学不动, best ep 9 早停; 论文 +13.5% 是 256ep+meta-tuned LR 的结论, 短训练协议不适用 | cosine decay 在 30ep 短训练协议下后期 LR 过低, best ep 9 早停, 负增益 (论文 +13.5% 需 256ep + meta-tuned LR); 本协议不适用 |
| C-12 | R17-realmlp-recipe | negative | Learnable scaling layer -0.0008: robust+clip 已归一化, 软特征选择无增量; lr×6 下早停 ep9 | robust+clip 已归一化, 软特征选择无增量; lr×6 早停 ep9, 负增益 (论文 +1.0% 未复现) |
| C-15 | R17-realmlp-recipe | negative | NT 参数化+数据驱动 init 灾难 -0.0375: NTK 改变梯度尺度, 与 LR=1e-3 不匹配 (论文配 lr=0.2+256ep+coslog4); best ep 28 持续爬升, 30ep 远不够 → 组件交互案例, NT 非独立组件 | NT 参数化改变梯度尺度与 LR=1e-3 不匹配 (论文配 lr=0.2+256ep+coslog4); best ep 28 持续爬升 30ep 远不够, 灾难性负增益; NT 非独立组件, 需与 LR 联合调 |
| P9-05 | R16-cancel-eventtime | negative | 原始 iat/burst 聚合负增益; 152 基线已含事件节奏 (o_*_near_far / t_*_gap / rowcount_near_far), 未条件化原始聚合为冗余噪声变体; 事件节奏信息存在须经 Z 式 market/tx 条件化 (Z 绿灯), 原始形式无增量 | compressed tabular baseline 已包含事件节奏信息; 原始聚合叠加有害 |

## none

| ID | 路线 | 证据 | 结论 | 原因 |
|---|---|---|---|---|
| E-01 | R07-state-conditioned | insufficient | gate 未过 (上界参考) | gate 未过 (上界参考) |
| P9-02 | R15-p9-quant | insufficient | 正信号: γ 单调 0→1, frozen 20月未触碰; 月度 12/20 正 (增益集中, 未达 70% gate) | 不重复: 在 PSEUDO 预测上再叠加无新意的 Z 集; 需与组件/融合叠加验证 |

## nonstationarity

| ID | 路线 | 证据 | 结论 | 原因 |
|---|---|---|---|---|
| E1-TW | R01-table-baseline | negative | 不再试任何时间衰减加权 | 全部无效或有害: 旧月份数据量价值更大, 漂移不是简单时间距离 |
| P4-07 | R09-hidden-information | negative | 不再做月度漂移预测模型 | 月度漂移不可预测 (无结构) |
| P9-03 | R15-p9-quant | negative | λ 全扫描 0.1-30: 仅 λ=3.0 单点 +0.0003, 两侧全负 ⇒ 噪声; V-REx 惩罚在 cosine+PSEUDO 下无可用区间 | 不再做 loss 方差惩罚类 (与 P4-07 漂移不可预测同向) |
| P8-02 | R14-o-to-t | negative | N1/N3/N5 均未形成可迁移时序 alpha | 跨月时序关系不足且不可复现 |

## not_identifiable

| ID | 路线 | 证据 | 结论 | 原因 |
|---|---|---|---|---|
| M-07 | R06-m-residual | not_identifiable | 截面结构不可用 | 截面结构不可用 |

## protocol_invalid

| ID | 路线 | 证据 | 结论 | 原因 |
|---|---|---|---|---|
| P10-01 | R11-scfi-z | invalid | 生产提交必须用与定标同源的模型 (同协议同 checkpoint 选择); 全量训练需留验证窗口定标 γ | scripts/p10_rq_prod_scfi.py scripts/p10_neutralize_prod.py |
| S-09 | R20-submissions | invalid | 格式错误 ref 后的最终 LB 显著低于 v8b | protocol_invalid：校准模型与提交模型历史范围不一致 |

## redundancy

| ID | 路线 | 证据 | 结论 | 原因 |
|---|---|---|---|---|
| P3-01 | R08-unsupervised-latent | negative | 不再做无监督 latent 直接进残差学习 | latent 与 152 手工特征重叠, 无新信息 |
| P9-06 | R16-cancel-eventtime | insufficient | L1/L2 DWI + trade entropy 边缘增 +0.0005~+0.0007, 方向稳定 12/20 月Δ正, 略偏 hi_act → YELLOW 进联合实验; DWOFI 仅 L1/L2 两层可构造, 熵与 t_buy_ratio 重叠, 与 P10-FM M1 L2 档位 (+0.0008 边缘) 一致 |  |
| P9-08 | R16-cancel-eventtime | negative | 撤单族归因闭合: cancel ⊂ Z — 替代否决 (A 只有 Z 一半强), 叠加否决 (J−Z −0.0008, 月度Δ正 10/20); Z 含 market/tx 条件化的 Z_ob_cancel_side_imb, 撤单族是 Z 绿灯的驱动成分但 Z 表达更好; 结论「直接用 A」不成立, 撤单族不做独立生产特征, 生产资产仍 152+73Z; A 的价值 = 归因证据 + Z 可解释成分 | raw 撤单侧拆叠加到 152+Z 无增量 (信息已被 Z 覆盖) |
