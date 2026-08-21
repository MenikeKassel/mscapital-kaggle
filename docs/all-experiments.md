# 全部实验（SSOT 生成视图）

> 本页由 `experiments/_tools/build_project_views.py` 生成。事实唯一来源是 `experiments/registry.csv`；`RESULTS.md` 仅为 append-only 历史日志。

| ID | 类型 | 路线 | 状态 | 证据 | 决策 | 分数 | Δ | 结论 |
|---|---|---|---|---|---|---|---|---|
| B0 | model | R01-table-baseline | superseded | superseded | NA | 0 / -0.007 | - | Δ=- → NA |
| B1 | feature | R01-table-baseline | completed | validated | GREEN | 0.130204 | - | Δ=- → GREEN |
| A1 | model | R01-table-baseline | completed | validated | GREEN | CV2 0.1338/CV3 0.1352/CV4 0.1418 | - | Δ=- → GREEN |
| A2 | feature | R01-table-baseline | completed | validated | GREEN | 0.1302 | +0.0333 | Δ=+0.0333 → GREEN |
| B1-LGO | ablation | R01-table-baseline | completed | validated | GREEN | 窗口-0.0102/盘口-0.0042/订单-0.0035/交叉-0.0003/成交-0.0000/EWM+0.0000 | - | Δ=- → GREEN |
| B2 | feature | R01-table-baseline | completed | validated | GREEN | 0.130237/0.129744 | -0.0000 | Δ=-0.0000 → GREEN |
| C1-FE | feature | R01-table-baseline | completed | negative | RED | 0.129760 | -0.0004 | 不再盲目堆窗口统计特征 |
| D1 | model | R01-table-baseline | completed | validated | YELLOW | 0.130763 | +0.0006 (leaves64) | Δ=+0.0006 (leaves64) → YELLOW |
| E1-TW | model | R01-table-baseline | completed | negative | RED | 0.125001~0.130161 | - | 不再试任何时间衰减加权 |
| F1 | feature | R01-table-baseline | completed | validated | YELLOW | 0.132065 | +0.0019 | loss 未收敛有空间 |
| F2 | model | R01-table-baseline | completed | validated | GREEN | 0.133736 | +0.0017 | Δ=+0.0017 → GREEN |
| G1 | ensemble | R01-table-baseline | completed | validated | GREEN | 0.137041 | +0.005 | Δ=+0.005 → GREEN |
| G2 | ensemble | R01-table-baseline | completed | validated | YELLOW | 0.137710 | +0.0007 | Δ=+0.0007 → YELLOW |
| G3 | ensemble | R01-table-baseline | completed | validated | GREEN | 0.138931 | +0.0012 | Δ=+0.0012 → GREEN |
| H1 | ensemble | R01-table-baseline | completed | negative | RED | 0.139332 | +0.0004 | 不再在表格特征+树+MLP 组合上做小增量融合 |
| P0-01 | protocol | R02-r2-drift | completed | validated | GREEN | MLP 7/7 垫底(0.1187), CatBoost 最稳健(0.1264), blend 每折全胜+0.003 | - | Δ=- → GREEN |
| P0-02 | protocol | R02-r2-drift | completed | validated | GREEN | AUC 0.733/0.777/0.766; m_sp_mean 三组 TOP1; m_book 漂移 45.1% | - | Δ=- → GREEN |
| P0-03 | model | R02-r2-drift | completed | negative | RED | 0.130015 | +0.0009 | 不再做表格模型权重微调 |
| P0-04 | protocol | R02-r2-drift | superseded | superseded | GREEN | T1 +0.0062 / T2 +0.0045 / T3 +0.0061 | +0.0050 平均 | 评估模型必须用 best-state 而非 last-epoch |
| P0-05 | model | R02-r2-drift | completed | validated | GREEN | R2 4/4 folds 全正 | +0.0023 mean | Δ=+0.0023 mean → GREEN |
| P0-06 | ensemble | R02-r2-drift | completed | validated | GREEN | T4 +0.0017 / PSEUDO +0.0015 | +0.0016 mean | Δ=+0.0016 mean → GREEN |
| S-04 | submission | R20-submissions | completed | validated | GREEN | - | - | Δ=- → GREEN |
| S-05 | submission | R20-submissions | completed | validated | GREEN | - | - | Δ=- → GREEN |
| S-07 | submission | R20-submissions | completed | validated | GREEN | blend 0.139683 (PSEUDO) | +0.010 (LB) | Δ=+0.010 (LB) → GREEN |
| S-08 | submission | R19-production-calibration | completed | validated | GREEN | - | +0.007 | Δ=+0.007 → GREEN |
| S-01 | submission | R20-submissions | completed | validated | GREEN | N/A | N/A | Δ=? → GREEN |
| S-02 | submission | R20-submissions | completed | validated | YELLOW | N/A | N/A | Δ=? → YELLOW |
| S-03 | submission | R20-submissions | completed | validated | YELLOW | N/A | N/A | Δ=? → YELLOW |
| S-06 | submission | R20-submissions | completed | negative | RED | N/A | N/A | 序列模型不通过 test 侧 corr 结构验证不得进生产融合 |
| P1-01 | build | R03-micro-primitives | completed | validated | NA | - | - | Δ=- → NA |
| P1-02 | feature | R03-micro-primitives | completed | validated | YELLOW | alpha mean +0.0022 | +0.0022 | drift 轴 ΔAUC +0.0168 ⚠️ 需第二轮归一化; 净效果为正 |
| P1-03 | ensemble | R03-micro-primitives | completed | validated | GREEN | T4 +0.0049 / PSEUDO +0.0036 | +0.0036 (PSEUDO) | Δ=+0.0036 (PSEUDO) → GREEN |
| P1-04 | feature | R03-micro-primitives | completed | negative | RED | T4 +0.0003 / PSEUDO -0.0012 | -0.0012 (PSEUDO) | 不再对微观特征做第二轮相对化 |
| P1-05 | feature | R05-sequence | completed | negative | RED | PSEUDO 0.0738 单模; 3-fold 融合全正 | +0.0035/+0.0023/+0.0044 | 不再用序列模型做生产融合 (除非 test 侧 corr 结构验证) |
| P2-01 | model | R04-realmlp-clean | completed | validated | GREEN | RealMLP 0.138560; v7 融合 0.139683 | +0.0048 | Δ=+0.0048 → GREEN |
| C-01 | ablation | R04-realmlp-clean | completed | validated | GREEN | ~0.143 (R61_70 复现 canonical 水准) | - | Δ=- → GREEN |
| C-02 | ablation | R04-realmlp-clean | completed | validated | GREEN | 30ep 全 outer 最优 | + | Δ=+ → GREEN |
| C-03 | ablation | R04-realmlp-clean | completed | validated | GREEN | PSEUDO 0.135051 | +0.000180 | Δ=+0.000180 → GREEN |
| C-04 | protocol | R04-realmlp-clean | completed | validated | GREEN | 0.142649/0.141762/0.143515/0.156924 | +0.005026 mean | 生产权重/尺度冻结后不得回改 (method/weight/scales/schema 全部冻结) |
| P3-01 | model | R08-unsupervised-latent | completed | negative | RED | PSEUDO -0.00076 | -0.00076 | 不再做无监督 latent 直接进残差学习 |
| P3-02 | model | R08-unsupervised-latent | completed | negative | RED | PSEUDO +0.00095 | +0.00095 | 不再用简单拼接做条件化 (FiLM 式融合留给完整版) |
| P3-03 | model | R08-unsupervised-latent | completed | negative | RED | corr 0.86~0.98 与现有特征 | - | 不再做掩码预训练 latent |
| P3-04 | model | R08-unsupervised-latent | completed | negative | RED | PSEUDO +0.0000 | +0.0000 | 不再做网格投影 |
| P3-05 | model | R08-unsupervised-latent | completed | negative | RED | \|r\|≤0.01 | - | 不投入完整 NHP 模型 |
| P4-01 | diagnostic | R09-hidden-information | completed | validated | GREEN | PSEUDO 正, AUC 增量 +0.019 | +0.019 AUC | Δ=+0.019 AUC → GREEN |
| P4-02 | diagnostic | R09-hidden-information | completed | validated | YELLOW | N/A | N/A | Δ=? → YELLOW |
| P4-03 | model | R09-hidden-information | completed | validated | YELLOW | unknown | unknown | Δ=? → YELLOW |
| P4-04 | protocol | R09-hidden-information | completed | validated | YELLOW | N/A | N/A | Δ=? → YELLOW |
| P4-05 | diagnostic | R09-hidden-information | completed | validated | YELLOW | y 与窗口内趋势相关, std∝波动率 | - | 公式未唯一确定 |
| P4-06 | diagnostic | R09-hidden-information | completed | validated | GREEN | ask 全空 → mid=0.5 计算假象 | - | Δ=- → GREEN |
| P4-07 | model | R09-hidden-information | completed | negative | RED | 决定性负结果 | - | 不再做月度漂移预测模型 |
| P4-08 | model | R09-hidden-information | completed | negative | RED | +0.0000 | +0.0000 | 不做残差均值直接预测 |
| P4-09 | model | R09-hidden-information | completed | validated | YELLOW | 分层证据 (高波动方向质量差) | - | Δ=- → YELLOW |
| P4-10 | ablation | R09-hidden-information | superseded | superseded | RED | 本地提升来自验证偏差 | - | 消融必须严格 nested, 不得重用验证期调参 |
| P4-11 | model | R09-hidden-information | completed | validated | YELLOW | unknown | unknown | Δ=? → YELLOW |
| P4-12 | ensemble | R09-hidden-information | completed | validated | YELLOW | unknown | unknown | Δ=? → YELLOW |
| P4-13 | model | R09-hidden-information | completed | validated | YELLOW | unknown | unknown | Δ=? → YELLOW |
| P4-14 | model | R09-hidden-information | completed | validated | YELLOW | unknown | unknown | Δ=? → YELLOW |
| P4-15 | diagnostic | R09-hidden-information | completed | negative | RED | 无资产身份痕迹 | - | 不再追资产身份 |
| P4-16 | diagnostic | R09-hidden-information | completed | validated | YELLOW | 分歧集中于高波动/低流动样本 | - | Δ=- → YELLOW |
| P4-17 | feature | R09-hidden-information | completed | validated | YELLOW | 弱正 | ~+0.0005 | Δ=~+0.0005 → YELLOW |
| M-01 | feature | R06-m-residual | completed | insufficient | RED | ~0 | ~0 | 事件流特征无残差增量 |
| M-02 | feature | R12-geometry-signature | completed | insufficient | RED | ~0 | ~0 | 几何特征无残差增量 |
| M-03 | feature | R12-geometry-signature | completed | insufficient | RED | unknown | unknown | 几何特征线整体关闭 |
| M-04 | feature | R12-geometry-signature | completed | insufficient | RED | ~0 | ~0 | 签名特征无残差增量 |
| M-05 | feature | R06-m-residual | completed | insufficient | RED | ~0 | ~0 | 交互特征无残差增量 |
| M-06 | feature | R06-m-residual | completed | insufficient | RED | ~0 | ~0 | 状态检索无残差增量 (P6R 前身) |
| M-07 | audit | R06-m-residual | completed | not_identifiable | RED | 无截面 alpha | - | 截面结构不可用 |
| E-01 | model | R07-state-conditioned | completed | insufficient | YELLOW | +0.0011 | +0.0011 | gate 未过 (上界参考) |
| E-02 | model | R07-state-conditioned | completed | descriptive | RED | 窗口内 cos 0.013, 跨月不可变现 | ~0 | 窗口内可解释 ≠ 跨月可预测增量 (负面链 #2) |
| E-03 | diagnostic | R07-state-conditioned | completed | descriptive | GREEN | 结论稳健 | - | Δ=- → GREEN |
| P5-01 | model | R11-scfi-z | completed | validated | YELLOW | 弱正 | ~+0.0005 | 序列信息大部分已在聚合特征中 |
| P5-02 | diagnostic | R11-scfi-z | completed | validated | GREEN | \|y\| corr 0.466 (幅度巨大); sign AUC 0.564 (方向微弱); corr(v7) 高 | - | Δ=- → GREEN |
| P5-03 | model | R11-scfi-z | completed | negative | RED | Δ_outer = -0.000146 | -0.000146 | 不再做简单 volatility/幅度 gate; 不再细扫 α∈[0,1] |
| P5-04 | model | R11-scfi-z | completed | validated | GREEN | C 臂 Δ=+0.0075 (17/20 月正, late +0.0091); blendΔ +0.00094 | +0.0075 | NN spot-check 无增益 (learner 依赖: LGB 特异) |
| P5-05 | model | R12-geometry-signature | completed | negative | RED | 全 ≤0.011 corr_y; R4 相位 -0.006 (反转) | - | 不再做 wavelet/shapelet/spectral CNN 短窗形态 |
| P5-06 | model | R11-scfi-z | completed | validated | GREEN | C/LGB +0.000849 (B1 +0.0013/B2 +0.0004) | +0.000849 | Δ=+0.000849 → GREEN |
| P5-07 | model | R11-scfi-z | completed | validated | GREEN | C 0.152570 (Δ+0.004044); blend w=0.50 Δ+0.001369; corr 0.940 | +0.004044 | SmallMLP 代理被推翻 (过弱) |
| P6-01 | model | R13-p6r-production | completed | validated | YELLOW | PSEUDO blendΔ +0.001435 (w=0.75); R61_70 +0.001369; std 比 0.9726 | +0.0014 | 未提交 (等用户拍板); 期望 LB 0.1423~0.1428 (v8b 已含 RealMLP 族, 增量压缩) |
| P6R-00 | model | R13-p6r-production | completed | negative | RED | 32/32 候选全正, 最大 PSEUDO Δ=+0.000588 (gate 39%) | +0.000588 | 不再做检索-残差预测路线 |
| P6R-01 | model | R13-p6r-production | planned | pending | NA | 未跑 | - | 挂起未执行 (用户未拍板) |
| P7-01 | model | R10-amplitude-gate | completed | negative | RED | ΔD = +0.00000 (α* 全 0.0) | +0.00000 | 不再做 volatility confidence calibration / Market→Confidence 假设 |
| C-05 | ablation | R17-realmlp-recipe | completed | validated | YELLOW | 0.113261 (PSEUDO eval 33-70, best epoch 12) | plain MLP 弱于 RQ 变体 (预期, E0 为干净锚点) | plain 3x256 MLP + robust+clip + 纯 MSE 基线成立; 选 checkpoint 指标 (cosine vs MSE) 在 PSEUDO fold 无差异 (同 epoch 12) |
| C-06 | ablation | R17-realmlp-recipe | completed | validated | YELLOW | 0.112136 (PSEUDO eval 33-70, best ep 19) | vs C-05 -0.001125 | 无 robust+clip (StandardScaler) 比 C-05 差 -0.0011 → 预处理组件有真实价值 (论文 reg +9.5% 方向一致) |
| C-07 | ablation | R17-realmlp-recipe | completed | validated | YELLOW | 0.114185 (PSEUDO eval 33-70, best ep 11) | vs C-05 +0.000923 | β2=0.95 小幅提升 +0.0009 (论文 +22.8% 量级未复现, 方向一致) |
| P8-01 | model | R14-o-to-t | completed | negative | RED | z(±1s)≈0; A_z 0/71月; placebo覆盖 | unknown | O→T 跨秒响应不存在; 同期lag=0是机械重合(0.33) |
| P9-01 | model | R15-p9-quant | completed | negative | RED | −0.0004~−0.0039 全 λ 负 | unknown | 机制性证伪: err_corr≈0.99 — 残差方向无信息(P5-03 AUC 0.51 已证), diversity 无米下锅 |
| P9-02 | model | R15-p9-quant | completed | insufficient | YELLOW | +0.001361 (frozen) | unknown | 正信号: γ 单调 0→1, frozen 20月未触碰; 月度 12/20 正 (增益集中, 未达 70% gate) |
| P9-03 | model | R15-p9-quant | completed | negative | RED | +0.0003 (孤峰) | unknown | λ 全扫描 0.1-30: 仅 λ=3.0 单点 +0.0003, 两侧全负 ⇒ 噪声; V-REx 惩罚在 cosine+PSEUDO 下无可用区间 |
| P10-01 | model | R11-scfi-z | completed | invalid | RED | +0.0005 (PSEUDO, 跨模型) | 0.116 (LB) | 生产提交必须用与定标同源的模型 (同协议同 checkpoint 选择); 全量训练需留验证窗口定标 γ |
| C-08 | ablation | R17-realmlp-recipe | completed | negative | RED | 0.112410 (PSEUDO eval 33-70, best ep 9) | vs C-05 -0.000851 | cosine decay (1→0) 在 30ep 短训练下为负 -0.0009: 后期 LR 过低学不动, best ep 9 早停; 论文 +13.5% 是 256ep+meta-tuned LR 的结论, 短训练协议不适用 |
| C-09 | ablation | R17-realmlp-recipe | completed | validated | GREEN | 0.122404 (PSEUDO eval 33-70, best ep 23) | vs C-05 +0.009143 | Parametric Mish 大增益 +0.0091 (论文 reg +4.8% 相对误差, 我们更强): ReLU 死区在 robust+clip 数据上代价高, 参数化激活修复它 |
| C-10 | ablation | R17-realmlp-recipe | completed | validated | GREEN | 0.125215 (PSEUDO eval 33-70, best ep 20) | vs C-05 +0.011954 | PL 数值嵌入最大增益 +0.0120 (论文最大组件 +20.6% 复现): 周期特征表达力强, 152→608 维 |
| C-11 | ablation | R17-realmlp-recipe | completed | validated | GREEN | 0.122602 (PSEUDO eval 33-70, best ep 23) | vs C-05 +0.009341 | PBLD +0.0093 略低于 PL 0.0026 (论文 PL≈PBLD 预期, 简化版参数更少) |
| C-12 | ablation | R17-realmlp-recipe | completed | negative | RED | 0.112446 (PSEUDO eval 33-70, best ep 9) | vs C-05 -0.000815 | Learnable scaling layer -0.0008: robust+clip 已归一化, 软特征选择无增量; lr×6 下早停 ep9 |
| C-13 | ablation | R17-realmlp-recipe | completed | validated | GREEN | 0.119053 (PSEUDO eval 33-70, best ep 14) | vs C-05 +0.005792 | dropout0.15+wd0.02 flat_cos 调度 +0.0058 (论文 constant 反而差 +3.6% 复现): 前稳后松的正则化节奏有效 |
| C-14 | ablation | R17-realmlp-recipe | completed | validated | GREEN | 0.115357 (PSEUDO eval 33-70, best ep 23) | vs C-05 +0.002096 | coslog4 周期调度 +0.0021 (论文 ns +0.4%, 我们更明显): 4 周期重启跳出局部最优, best ep 23 持续学习 |
| C-15 | ablation | R17-realmlp-recipe | completed | negative | RED | 0.075803 (PSEUDO eval 33-70, best ep 28) | vs C-05 -0.037458 | NT 参数化+数据驱动 init 灾难 -0.0375: NTK 改变梯度尺度, 与 LR=1e-3 不匹配 (论文配 lr=0.2+256ep+coslog4); best ep 28 持续爬升, 30ep 远不够 → 组件交互案例, NT 非独立组件 |
| P9-04 | model | R16-cancel-eventtime | completed | validated | YELLOW | 0.118510 (feat frozen) | +0.004146 frozen / +0.004378 eval | 侧拆撤单不对称(流动性撤退)真实有效: +0.0041 frozen / 20-20 聚合月正 / 13-20 月Δ正, 但 regime 集中 (hi_act +0.0103 / low_act -0.0064) → 非 clean GREEN, 进联合/校准验证; 与 Z 绿灯同源 (Z_ob_cancel_side_imb) 互为归因 |
| P9-05 | model | R16-cancel-eventtime | completed | negative | RED | 0.110477 (feat frozen) | -0.003887 frozen / -0.002754 eval | 原始 iat/burst 聚合负增益; 152 基线已含事件节奏 (o_*_near_far / t_*_gap / rowcount_near_far), 未条件化原始聚合为冗余噪声变体; 事件节奏信息存在须经 Z 式 market/tx 条件化 (Z 绿灯), 原始形式无增量 |
| P9-06 | model | R16-cancel-eventtime | completed | insufficient | YELLOW | 0.114906 (feat frozen) | +0.000542 frozen / +0.000704 eval | L1/L2 DWI + trade entropy 边缘增 +0.0005~+0.0007, 方向稳定 12/20 月Δ正, 略偏 hi_act → YELLOW 进联合实验; DWOFI 仅 L1/L2 两层可构造, 熵与 t_buy_ratio 重叠, 与 P10-FM M1 L2 档位 (+0.0008 边缘) 一致 |
| P9-07 | model | R16-cancel-eventtime | completed | validated | GREEN | raw mean +0.007148 | raw 3-seed +0.004146/+0.005986/+0.011313 (3/3 正) | 撤单侧拆不对称稳健 GREEN 候选: 3-seed 均值 frozen +0.0071 (3/3 正); 单 seed 2026 的 low_act 拖累 (-0.0064) 是 seed 噪声 (2027/28 low_act +0.0105/+0.0073), 双 regime 均值皆正; mask(低撤单置0) 三 seed 均劣于 raw → 机制猜想证伪, hard-mask 路径不重复; 与 Z 同源 (Z_ob_cancel_side_imb) 互为归因, 是 Z 绿灯候选主成分 |
| P9-08 | model | R16-cancel-eventtime | completed | negative | RED | Z 0.123030 (+0.008666) | cancel +0.004146; Z +0.008666; cancel+Z +0.007852 (J−Z −0.0008) | 撤单族归因闭合: cancel ⊂ Z — 替代否决 (A 只有 Z 一半强), 叠加否决 (J−Z −0.0008, 月度Δ正 10/20); Z 含 market/tx 条件化的 Z_ob_cancel_side_imb, 撤单族是 Z 绿灯的驱动成分但 Z 表达更好; 结论「直接用 A」不成立, 撤单族不做独立生产特征, 生产资产仍 152+73Z; A 的价值 = 归因证据 + Z 可解释成分 |
| P6-02 | protocol | R13-p6r-production | completed | validated | YELLOW | RealMLP-C 0.139248; blendΔ +0.001460 (w=0.70 tuned 21-32) | +0.001460 | 重跑 Z 线生产协议: RealMLP-C 152+73Z 同源校准 + test 门禁全部复现 (standalone 0.139248/canon 0.142550/std ratio 0.9726 与 8/15 逐位一致, 确定性); blendΔ 今日 +0.001460 略高于 8/15 +0.001435; 三窗口 w 敏感性峰值 51-60 w=0.55 / 61-70 w=0.65 / PSEUDO w=0.55 → 生产 blend w=0.55; test 侧 corr(RealMLP-C,v8b)=0.9280 结构一致 |
| P11-01 | diagnostic | R18-blsm | completed | descriptive | GREEN | 行为特征 8PC 累计解释 0.60; 4 代理(order_count/txn_count/vol)解释 PC R2≈0 (−0.0006~−0.0009) | 行为 latent 独立于 activity/volatility/month (month AUC≈0.5); PC4 rankIC −0.0036 (p<5e-5) | BLSM 行为状态结构存在 (EXIST): 27 特征 PCA↘8 维; 关键证据=现有活动度/波动/计数代理几乎无法解释行为 latent (R2≈0), 不被月主导; 仅 PC4 与 target 显著相关 (IC −0.0036, 量级弱)。行为状态存在但不强 → 进 G1 (incremental gate) 回答是否有基线之上增量 |
| P8-02 | model | R14-o-to-t | completed | negative | RED | unknown | unknown | N1/N3/N5 均未形成可迁移时序 alpha |
| P10-02 | model | R11-scfi-z | completed | validated | GREEN | 0.143 | +0.0075 | SCFI plain realization 提供条件增量 |
| P10-03 | model | R11-scfi-z | completed | validated | GREEN | 0.1438 | +0.0040 | RQ 条件表示在本地验证有增量 |
| P10-04 | feature | R11-scfi-z | completed | validated | GREEN | unknown | +0.0060 | 二阶条件项进入 152+73Z 冻结资产 |
| P10-05 | model | R11-scfi-z | completed | validated | YELLOW | unknown | unknown | L2 变体有局部信息但未形成独立候选 |
| P10-06 | model | R11-scfi-z | completed | validated | YELLOW | unknown | unknown | 事件跳变条件项未超过已冻结 Z |
| P10-07 | model | R11-scfi-z | completed | validated | YELLOW | unknown | unknown | 成交条件 H1/H2/H3 只作诊断，不独立生产 |
| P6-03 | build | R13-p6r-production | completed | validated | GREEN | +0.001 | +0.001 | 纯原创 v5 Table + RealMLP-C 融合构建完成 |
| S-09 | submission | R20-submissions | completed | invalid | RED | 0.1438 | N/A | 格式错误 ref 后的最终 LB 显著低于 v8b |
| S-10 | submission | R20-submissions | completed | validated | YELLOW | N/A | N/A | Z 线重跑低于 external lb142 但高于纯原创锚点 |
| S-11 | submission | R20-submissions | completed | validated | GREEN | N/A | N/A | 纯原创基准提交，作为 self-owned anchor |
