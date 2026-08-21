# 实验时间线

> 每条谱系只列一次；详细字段见 `all-experiments.md`。

| 日期 | ID | 路线 | 证据 | 结论 |
|---|---|---|---|---|
| 2026-08-10 | A1 | R01-table-baseline | validated | Δ=- → GREEN |
| 2026-08-10 | A2 | R01-table-baseline | validated | Δ=+0.0333 → GREEN |
| 2026-08-10 | B0 | R01-table-baseline | superseded | Δ=- → NA |
| 2026-08-10 | B1 | R01-table-baseline | validated | Δ=- → GREEN |
| 2026-08-10 | B1-LGO | R01-table-baseline | validated | Δ=- → GREEN |
| 2026-08-10 | B2 | R01-table-baseline | validated | Δ=-0.0000 → GREEN |
| 2026-08-10 | C1-FE | R01-table-baseline | negative | 不再盲目堆窗口统计特征 |
| 2026-08-10 | D1 | R01-table-baseline | validated | Δ=+0.0006 (leaves64) → YELLOW |
| 2026-08-10 | E1-TW | R01-table-baseline | negative | 不再试任何时间衰减加权 |
| 2026-08-10 | F1 | R01-table-baseline | validated | loss 未收敛有空间 |
| 2026-08-10 | F2 | R01-table-baseline | validated | Δ=+0.0017 → GREEN |
| 2026-08-10 | G1 | R01-table-baseline | validated | Δ=+0.005 → GREEN |
| 2026-08-10 | G2 | R01-table-baseline | validated | Δ=+0.0007 → YELLOW |
| 2026-08-10 | G3 | R01-table-baseline | validated | Δ=+0.0012 → GREEN |
| 2026-08-11 | H1 | R01-table-baseline | negative | 不再在表格特征+树+MLP 组合上做小增量融合 |
| 2026-08-11 | P0-01 | R02-r2-drift | validated | Δ=- → GREEN |
| 2026-08-11 | P0-02 | R02-r2-drift | validated | Δ=- → GREEN |
| 2026-08-11 | P0-03 | R02-r2-drift | negative | 不再做表格模型权重微调 |
| 2026-08-11 | P0-04 | R02-r2-drift | superseded | 评估模型必须用 best-state 而非 last-epoch |
| 2026-08-11 | P0-05 | R02-r2-drift | validated | Δ=+0.0023 mean → GREEN |
| 2026-08-11 | P0-06 | R02-r2-drift | validated | Δ=+0.0016 mean → GREEN |
| 2026-08-11 | P1-01 | R03-micro-primitives | validated | Δ=- → NA |
| 2026-08-11 | P1-02 | R03-micro-primitives | validated | drift 轴 ΔAUC +0.0168 ⚠️ 需第二轮归一化; 净效果为正 |
| 2026-08-11 | P1-03 | R03-micro-primitives | validated | Δ=+0.0036 (PSEUDO) → GREEN |
| 2026-08-11 | P1-04 | R03-micro-primitives | negative | 不再对微观特征做第二轮相对化 |
| 2026-08-11 | P1-05 | R05-sequence | negative | 不再用序列模型做生产融合 (除非 test 侧 corr 结构验证) |
| 2026-08-11 | S-04 | R20-submissions | validated | Δ=- → GREEN |
| 2026-08-11 | S-05 | R20-submissions | validated | Δ=- → GREEN |
| 2026-08-12 | P2-01 | R04-realmlp-clean | validated | Δ=+0.0048 → GREEN |
| 2026-08-12 | S-07 | R20-submissions | validated | Δ=+0.010 (LB) → GREEN |
| 2026-08-12 | S-08 | R19-production-calibration | validated | Δ=+0.007 → GREEN |
| 2026-08-13 | C-01 | R04-realmlp-clean | validated | Δ=- → GREEN |
| 2026-08-13 | C-02 | R04-realmlp-clean | validated | Δ=+ → GREEN |
| 2026-08-13 | C-03 | R04-realmlp-clean | validated | Δ=+0.000180 → GREEN |
| 2026-08-13 | C-04 | R04-realmlp-clean | validated | 生产权重/尺度冻结后不得回改 (method/weight/scales/schema 全部冻结) |
| 2026-08-13 | E-01 | R07-state-conditioned | insufficient | gate 未过 (上界参考) |
| 2026-08-13 | E-02 | R07-state-conditioned | descriptive | 窗口内可解释 ≠ 跨月可预测增量 (负面链 #2) |
| 2026-08-13 | E-03 | R07-state-conditioned | descriptive | Δ=- → GREEN |
| 2026-08-13 | M-01 | R06-m-residual | insufficient | 事件流特征无残差增量 |
| 2026-08-13 | M-02 | R12-geometry-signature | insufficient | 几何特征无残差增量 |
| 2026-08-13 | M-04 | R12-geometry-signature | insufficient | 签名特征无残差增量 |
| 2026-08-13 | M-05 | R06-m-residual | insufficient | 交互特征无残差增量 |
| 2026-08-13 | M-06 | R06-m-residual | insufficient | 状态检索无残差增量 (P6R 前身) |
| 2026-08-13 | M-07 | R06-m-residual | not_identifiable | 截面结构不可用 |
| 2026-08-14 | P3-01 | R08-unsupervised-latent | negative | 不再做无监督 latent 直接进残差学习 |
| 2026-08-14 | P3-02 | R08-unsupervised-latent | negative | 不再用简单拼接做条件化 (FiLM 式融合留给完整版) |
| 2026-08-14 | P3-03 | R08-unsupervised-latent | negative | 不再做掩码预训练 latent |
| 2026-08-14 | P3-04 | R08-unsupervised-latent | negative | 不再做网格投影 |
| 2026-08-14 | P3-05 | R08-unsupervised-latent | negative | 不投入完整 NHP 模型 |
| 2026-08-14 | P4-01 | R09-hidden-information | validated | Δ=+0.019 AUC → GREEN |
| 2026-08-14 | P4-05 | R09-hidden-information | validated | 公式未唯一确定 |
| 2026-08-14 | P4-06 | R09-hidden-information | validated | Δ=- → GREEN |
| 2026-08-14 | P4-07 | R09-hidden-information | negative | 不再做月度漂移预测模型 |
| 2026-08-14 | P4-08 | R09-hidden-information | negative | 不做残差均值直接预测 |
| 2026-08-14 | P4-09 | R09-hidden-information | validated | Δ=- → YELLOW |
| 2026-08-14 | P4-10 | R09-hidden-information | superseded | 消融必须严格 nested, 不得重用验证期调参 |
| 2026-08-14 | P4-15 | R09-hidden-information | negative | 不再追资产身份 |
| 2026-08-14 | P4-16 | R09-hidden-information | validated | Δ=- → YELLOW |
| 2026-08-14 | P4-17 | R09-hidden-information | validated | Δ=~+0.0005 → YELLOW |
| 2026-08-14 | P5-01 | R11-scfi-z | validated | 序列信息大部分已在聚合特征中 |
| 2026-08-14 | P5-02 | R11-scfi-z | validated | Δ=- → GREEN |
| 2026-08-14 | P5-03 | R11-scfi-z | negative | 不再做简单 volatility/幅度 gate; 不再细扫 α∈[0,1] |
| 2026-08-14 | P5-04 | R11-scfi-z | validated | NN spot-check 无增益 (learner 依赖: LGB 特异) |
| 2026-08-14 | P5-05 | R12-geometry-signature | negative | 不再做 wavelet/shapelet/spectral CNN 短窗形态 |
| 2026-08-14 | P6R-00 | R13-p6r-production | negative | 不再做检索-残差预测路线 |
| 2026-08-15 | C-05 | R17-realmlp-recipe | validated | plain 3x256 MLP + robust+clip + 纯 MSE 基线成立; 选 checkpoint 指标 (cosine vs MSE) 在 PSEUDO fold 无差异 (同 epoch 12) |
| 2026-08-15 | C-06 | R17-realmlp-recipe | validated | 无 robust+clip (StandardScaler) 比 C-05 差 -0.0011 → 预处理组件有真实价值 (论文 reg +9.5% 方向一致) |
| 2026-08-15 | C-07 | R17-realmlp-recipe | validated | β2=0.95 小幅提升 +0.0009 (论文 +22.8% 量级未复现, 方向一致) |
| 2026-08-15 | M-03 | R12-geometry-signature | insufficient | 几何特征线整体关闭 |
| 2026-08-15 | P4-02 | R09-hidden-information | validated | Δ=? → YELLOW |
| 2026-08-15 | P4-03 | R09-hidden-information | validated | Δ=? → YELLOW |
| 2026-08-15 | P4-04 | R09-hidden-information | validated | Δ=? → YELLOW |
| 2026-08-15 | P4-11 | R09-hidden-information | validated | Δ=? → YELLOW |
| 2026-08-15 | P4-12 | R09-hidden-information | validated | Δ=? → YELLOW |
| 2026-08-15 | P4-13 | R09-hidden-information | validated | Δ=? → YELLOW |
| 2026-08-15 | P4-14 | R09-hidden-information | validated | Δ=? → YELLOW |
| 2026-08-15 | P5-06 | R11-scfi-z | validated | Δ=+0.000849 → GREEN |
| 2026-08-15 | P5-07 | R11-scfi-z | validated | SmallMLP 代理被推翻 (过弱) |
| 2026-08-15 | P6-01 | R13-p6r-production | validated | 未提交 (等用户拍板); 期望 LB 0.1423~0.1428 (v8b 已含 RealMLP 族, 增量压缩) |
| 2026-08-15 | P6R-01 | R13-p6r-production | pending | 挂起未执行 (用户未拍板) |
| 2026-08-15 | P7-01 | R10-amplitude-gate | negative | 不再做 volatility confidence calibration / Market→Confidence 假设 |
| 2026-08-15 | P8-01 | R14-o-to-t | negative | O→T 跨秒响应不存在; 同期lag=0是机械重合(0.33) |
| 2026-08-15 | P9-01 | R15-p9-quant | negative | 机制性证伪: err_corr≈0.99 — 残差方向无信息(P5-03 AUC 0.51 已证), diversity 无米下锅 |
| 2026-08-15 | P9-02 | R15-p9-quant | insufficient | 正信号: γ 单调 0→1, frozen 20月未触碰; 月度 12/20 正 (增益集中, 未达 70% gate) |
| 2026-08-15 | P9-03 | R15-p9-quant | negative | λ 全扫描 0.1-30: 仅 λ=3.0 单点 +0.0003, 两侧全负 ⇒ 噪声; V-REx 惩罚在 cosine+PSEUDO 下无可用区间 |
| 2026-08-15 | S-01 | R20-submissions | validated | Δ=? → GREEN |
| 2026-08-15 | S-02 | R20-submissions | validated | Δ=? → YELLOW |
| 2026-08-15 | S-03 | R20-submissions | validated | Δ=? → YELLOW |
| 2026-08-15 | S-06 | R20-submissions | negative | 序列模型不通过 test 侧 corr 结构验证不得进生产融合 |
| 2026-08-16 | C-08 | R17-realmlp-recipe | negative | cosine decay (1→0) 在 30ep 短训练下为负 -0.0009: 后期 LR 过低学不动, best ep 9 早停; 论文 +13.5% 是 256ep+meta-tuned LR 的结论, 短训练协议不适用 |
| 2026-08-16 | C-09 | R17-realmlp-recipe | validated | Parametric Mish 大增益 +0.0091 (论文 reg +4.8% 相对误差, 我们更强): ReLU 死区在 robust+clip 数据上代价高, 参数化激活修复它 |
| 2026-08-16 | C-10 | R17-realmlp-recipe | validated | PL 数值嵌入最大增益 +0.0120 (论文最大组件 +20.6% 复现): 周期特征表达力强, 152→608 维 |
| 2026-08-16 | C-11 | R17-realmlp-recipe | validated | PBLD +0.0093 略低于 PL 0.0026 (论文 PL≈PBLD 预期, 简化版参数更少) |
| 2026-08-16 | C-12 | R17-realmlp-recipe | negative | Learnable scaling layer -0.0008: robust+clip 已归一化, 软特征选择无增量; lr×6 下早停 ep9 |
| 2026-08-16 | C-13 | R17-realmlp-recipe | validated | dropout0.15+wd0.02 flat_cos 调度 +0.0058 (论文 constant 反而差 +3.6% 复现): 前稳后松的正则化节奏有效 |
| 2026-08-16 | C-14 | R17-realmlp-recipe | validated | coslog4 周期调度 +0.0021 (论文 ns +0.4%, 我们更明显): 4 周期重启跳出局部最优, best ep 23 持续学习 |
| 2026-08-16 | C-15 | R17-realmlp-recipe | negative | NT 参数化+数据驱动 init 灾难 -0.0375: NTK 改变梯度尺度, 与 LR=1e-3 不匹配 (论文配 lr=0.2+256ep+coslog4); best ep 28 持续爬升, 30ep 远不够 → 组件交互案例, NT 非独立组件 |
| 2026-08-16 | P10-01 | R11-scfi-z | invalid | 生产提交必须用与定标同源的模型 (同协议同 checkpoint 选择); 全量训练需留验证窗口定标 γ |
| 2026-08-20 | P9-04 | R16-cancel-eventtime | validated | 侧拆撤单不对称(流动性撤退)真实有效: +0.0041 frozen / 20-20 聚合月正 / 13-20 月Δ正, 但 regime 集中 (hi_act +0.0103 / low_act -0.0064) → 非 clean GREEN, 进联合/校准验证; 与 Z 绿灯同源 (Z_ob_cancel_side_imb) 互为归因 |
| 2026-08-20 | P9-05 | R16-cancel-eventtime | negative | 原始 iat/burst 聚合负增益; 152 基线已含事件节奏 (o_*_near_far / t_*_gap / rowcount_near_far), 未条件化原始聚合为冗余噪声变体; 事件节奏信息存在须经 Z 式 market/tx 条件化 (Z 绿灯), 原始形式无增量 |
| 2026-08-20 | P9-06 | R16-cancel-eventtime | insufficient | L1/L2 DWI + trade entropy 边缘增 +0.0005~+0.0007, 方向稳定 12/20 月Δ正, 略偏 hi_act → YELLOW 进联合实验; DWOFI 仅 L1/L2 两层可构造, 熵与 t_buy_ratio 重叠, 与 P10-FM M1 L2 档位 (+0.0008 边缘) 一致 |
| 2026-08-20 | P9-07 | R16-cancel-eventtime | validated | 撤单侧拆不对称稳健 GREEN 候选: 3-seed 均值 frozen +0.0071 (3/3 正); 单 seed 2026 的 low_act 拖累 (-0.0064) 是 seed 噪声 (2027/28 low_act +0.0105/+0.0073), 双 regime 均值皆正; mask(低撤单置0) 三 seed 均劣于 raw → 机制猜想证伪, hard-mask 路径不重复; 与 Z 同源 (Z_ob_cancel_side_imb) 互为归因, 是 Z 绿灯候选主成分 |
| 2026-08-20 | P9-08 | R16-cancel-eventtime | negative | 撤单族归因闭合: cancel ⊂ Z — 替代否决 (A 只有 Z 一半强), 叠加否决 (J−Z −0.0008, 月度Δ正 10/20); Z 含 market/tx 条件化的 Z_ob_cancel_side_imb, 撤单族是 Z 绿灯的驱动成分但 Z 表达更好; 结论「直接用 A」不成立, 撤单族不做独立生产特征, 生产资产仍 152+73Z; A 的价值 = 归因证据 + Z 可解释成分 |
| 2026-08-21 | P10-02 | R11-scfi-z | validated | SCFI plain realization 提供条件增量 |
| 2026-08-21 | P10-03 | R11-scfi-z | validated | RQ 条件表示在本地验证有增量 |
| 2026-08-21 | P10-04 | R11-scfi-z | validated | 二阶条件项进入 152+73Z 冻结资产 |
| 2026-08-21 | P10-05 | R11-scfi-z | validated | L2 变体有局部信息但未形成独立候选 |
| 2026-08-21 | P10-06 | R11-scfi-z | validated | 事件跳变条件项未超过已冻结 Z |
| 2026-08-21 | P10-07 | R11-scfi-z | validated | 成交条件 H1/H2/H3 只作诊断，不独立生产 |
| 2026-08-21 | P11-01 | R18-blsm | descriptive | BLSM 行为状态结构存在 (EXIST): 27 特征 PCA↘8 维; 关键证据=现有活动度/波动/计数代理几乎无法解释行为 latent (R2≈0), 不被月主导; 仅 PC4 与 target 显著相关 (IC −0.0036, 量级弱)。行为状态存在但不强 → 进 G1 (incremental gate) 回答是否有基线之上增量 |
| 2026-08-21 | P6-02 | R13-p6r-production | validated | 重跑 Z 线生产协议: RealMLP-C 152+73Z 同源校准 + test 门禁全部复现 (standalone 0.139248/canon 0.142550/std ratio 0.9726 与 8/15 逐位一致, 确定性); blendΔ 今日 +0.001460 略高于 8/15 +0.001435; 三窗口 w 敏感性峰值 51-60 w=0.55 / 61-70 w=0.65 / PSEUDO w=0.55 → 生产 blend w=0.55; test 侧 corr(RealMLP-C,v8b)=0.9280 结构一致 |
| 2026-08-21 | P6-03 | R13-p6r-production | validated | 纯原创 v5 Table + RealMLP-C 融合构建完成 |
| 2026-08-21 | P8-02 | R14-o-to-t | negative | N1/N3/N5 均未形成可迁移时序 alpha |
| 2026-08-21 | S-09 | R20-submissions | invalid | 格式错误 ref 后的最终 LB 显著低于 v8b |
| 2026-08-21 | S-10 | R20-submissions | validated | Z 线重跑低于 external lb142 但高于纯原创锚点 |
| 2026-08-21 | S-11 | R20-submissions | validated | 纯原创基准提交，作为 self-owned anchor |
