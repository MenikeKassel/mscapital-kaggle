# MSCapital 实验索引 (Experiment Index)

> **研究历史总入口** — 5 分钟内看懂整个项目演化。
> 权威链: 本文件 → `experiments/registry.csv` (机器可读) → 各实验 `experiments/<phase>/<ID>/README.md` → 阶段报告 (docs/)
> 更新: 2026-08-15 (仓库工程化整理 Phase H)

---

## 当前状态速览

| 项 | 值 |
|---|---|
| 当前最好 Public LB | **0.142** (#30/107), v8b = v7 + 外部 lb142 推理包 (0.5/0.5) |
| 最佳自研单模型 | RealMLP (152 特征, 复刻公开方案), 单模型 PSEUDO 0.138560 |
| 当前生产基线 | C-04 frozen: `0.63×RMS(RealMLP) + 0.37×RMS(Clean Table)` (0.142649/0.141762/0.143515/0.156924) |
| 冻结验证纪律 | 新候选必须过: 双窗口 (R61_70 + PSEUDO) blendΔ 一致 + 分布门禁 (std 比≈1) |
| 待拍板 | ① P6 提交候选 (v8b+0.55×RealMLP-C, 期望 LB 0.1423~0.1428) ② P6R-01 终裁 ③ 下一步 O→T lag response |
| 活跃方案 | plan-v1.9.0.md (唯一 current source of truth) |

---

## Baseline 表格阶梯 (2026-08-10/11)

| ID | 实验 | 结果 | 状态 | 核心结论 |
| -- | -- | -: | -- | ---- |
| B0 | 常数基线 sanity | 0 / -0.007 | SUPERSEDED | 社区数据 sanity |
| B1 | LightGBM 官方复刻 | CV 0.130204 | GREEN | 锚点; LB 0.117 |
| A1 | CV 切分敏感性 | 0.1302~0.1418 | GREEN | CV1 最诚实 |
| A2 | 特征数 17 vs 90 | +0.0333 | GREEN | 增量特征是主力 |
| B1-LGO | 特征组消融 | 窗口 -0.0102 | GREEN | 窗口统计绝对主力 |
| B2 | 特征精简 | -0.0000 | GREEN | 84 可采纳 |
| C1-FE | 增强窗口特征 | -0.0004 | RED | **特征工程到顶 (90 甜点位)** |
| D1 | 超参单变量 | +0.0006 | YELLOW | leaves64 采纳 |
| E1-TW | 时间衰减加权 | 全无效 | RED | **不再试时间衰减** |
| F1 | 轻量 MLP | +0.0019 | YELLOW | 表格 NN 温和超越 |
| G1/G2/G3 | 融合链 | 0.1370→0.1389 | GREEN | 融合是最大杠杆 |
| H1 | 五模型 | +0.0004 | RED | **表格路线 0.122 平台期** |

提交: **v1 0.122** (#79, 来自 G3) → v2 0.122 (H1) → v3 0.122 (P0-03) → **v4 0.123** (S-04) — 完整登记在 [submissions/README.md](../submissions/README.md)

## P0 Protocol 验证 (2026-08-11)

| ID | 实验 | 结果 | 状态 | 核心结论 |
| -- | -- | -: | -- | ---- |
| P0-01 | Temporal Matrix | MLP 7/7 垫底 | GREEN | CV1 模型选择不可信; CatBoost 最稳健 |
| P0-02 | Adversarial Validation | AUC 0.73-0.78 | GREEN | **预测主力=漂移主力**; 漂移机制解释 |
| P0-03 | 权重重估 | +0.0009 | RED | 权重微调无法突破平台期 |
| P0-04 | MLP Fairness | +0.0050 | SUPERSEDED | last-epoch 评估 bug 修正 |
| P0-05 | R2 归一化干预 | +0.0023 (4/4) | GREEN | **归一化延长 alpha 寿命** |
| P0-06 | R2 × 全融合 | +0.0016 | GREEN | 兑现 |

提交: **v4 0.123** (S-04, R2+temporal, 科研闭环验证成功)

## P1 表示与序列 (2026-08-11/12)

| ID | 实验 | 结果 | 状态 | 核心结论 |
| -- | -- | -: | -- | ---- |
| P1-01 | 22 微观 primitive | 构建完成 | GREEN | 事件流微观结构特征 |
| P1-02 | 双轴筛选 | alpha +0.0022 | YELLOW | 部分特征携带漂移 ⚠️ |
| P1-03 | 全融合 (R2+micro) | PSEUDO +0.0036 | GREEN | 迄今最大单步提升 |
| P1-04 | 相对化第二轮 | PSEUDO -0.0012 | RED | 第一轮已最优 |
| P1-02 | TCN 双塔 | 融合 3 折全正 | RED | **test 分布外灾难 (v6 0.082)**; 序列模型不可信 |

提交: **v5 0.125** (S-05, R2+22微观) → **v6 0.082** (TCN, N005 灾难) → **v7 0.135** (S-07, RealMLP 复刻 0.8/0.2) → **v8a 0.139** → **v8b 0.142** (S-08, #30, +lb142 0.5/0.5)

## P2 校准 (2026-08-12, Codex)

| ID | 实验 | 结果 | 状态 | 核心结论 |
| -- | -- | -: | -- | ---- |
| P2 | RealMLP PSEUDO 定标 | 0.138560; v7 融合 0.139683 | GREEN | Regime B gap 0.0047; 尺度门禁 0.7447 |

## C 系列 Clean Baseline v2 (2026-08-13)

| ID | 实验 | 结果 | 状态 | 核心结论 |
| -- | -- | -: | -- | ---- |
| C-01 | Clean RealMLP-v2a | ~0.143 | GREEN | 干净复刻基线 |
| C-02 | RealMLP 消融 | 30ep 最优 | GREEN | 30-epoch 生产方案 |
| C-03 | Clean Table v2 | PSEUDO 0.135051 | GREEN | 与 legacy v5 一致 (+0.00018) |
| C-04 | Clean Baseline v2 冻结 | +0.0050 mean | GREEN | **生产规则冻结**; 不得回改 |

## P3 下一代方法 (2026-08-14) — 全部 gate F

| ID | 实验 | 结果 | 状态 | 核心结论 |
| -- | -- | -: | -- | ---- |
| P3-01 | SAE | PSEUDO -0.00076 | RED | latent 与 152 重叠 |
| P3-02 | 状态条件化 | +0.00095 | RED | 拼接稀释 E-01 信号 |
| P3-03 | TinyLOBERT 掩码 | corr 0.86~0.98 | RED | 掩码 latent 无独立信息 |
| P3-04 | 2.5D 网格 | +0.0000 | RED | 投影无信号 |
| P3-05 | NHP 强度 | \|r\|≤0.01 | RED | 不投入完整 NHP |

## P4 隐藏信息调查 (2026-08-13/14)

| ID | 实验 | 结果 | 状态 | 核心结论 |
| -- | -- | -: | -- | ---- |
| P4-01a | 600s 长上下文 | AUC +0.019 | GREEN | **H4 升级为直接证据** (600s 流有独立信息) |
| P4-02 | factors/forms/OFI | OFI 弱正 | YELLOW | 部分线索 |
| P4-03 | target 逆向 | 趋势相关 | YELLOW | 公式未唯一确定 |
| P4-04 | 0.5 价簇取证 | ask 全空假象 | GREEN | 非第二 instrument (示例报告确认) |
| P4-05 | 月度漂移预测 | 决定性负结果 | RED | 漂移不可预测 |
| P4-08 | 长残差 | +0.0000 | RED | 负面链 #1 |
| P4-07 | halfgroup 分层 | 分层证据 | YELLOW | 高波动方向质量差 |
| P4-10 | loss ablation | 验证偏差 | INVALID | **消融必须严格 nested** |
| P4-15 | 资产/时间身份 | 无证据 | RED | H1+H2 关闭 |
| P4-16 | 分歧取证 | 高波动集中 | YELLOW | 外部模型信息面线索 |
| P4-17 | market history 特征 | ~+0.0005 | YELLOW | 600s 特征弱正 |

## M 系列残差表示 (2026-08-13) — 全部 RED

| ID | 实验 | 结果 | 状态 | 核心结论 |
| -- | -- | -: | -- | ---- |
| M-01 | Event Flow | ~0 | RED | 无残差增量 |
| M-02/M-03 | LOB 几何 | ~0 | RED | 无残差增量 |
| M-04 | Path Signature | ~0 | RED | 无残差增量 |
| M-05 | Optiver 交互 | ~0 | RED | 无残差增量 |
| M-06 | 状态 KNN | ~0 | RED | 无残差增量 (P6R 前身) |
| M-07 | 截面审计 | 无截面 alpha | RED | 截面不可用 |

## E 系列状态条件化 (2026-08-13)

| ID | 实验 | 结果 | 状态 | 核心结论 |
| -- | -- | -: | -- | ---- |
| E-01 | ReVol-lite | +0.0011 | YELLOW | 未过 gate (上界参考) |
| E-02 | Reconditionor-lite | 窗口内 cos 0.013 | RED | **窗口内可解释 ≠ 跨月可预测** (负面链 #2) |
| E-03 | 稳定性审计 | 稳健 | GREEN | 结论稳健 |

## P5 市场探针 (2026-08-14/15)

| ID | 实验 | 结果 | 状态 | 核心结论 |
| -- | -- | -: | -- | ---- |
| P5-01 | market-only 审判 | ~+0.0005 | YELLOW | 序列信息大部分已在聚合特征 |
| P5-02 | 信息审计 (五条实锤) | \|y\| corr **0.466** | GREEN | **幅度富矿; 方向微弱 (AUC 0.564)**; GPT1 猜想反转 |
| P5-03 | MAG-Gate | -0.000146 | RED | **幅度门控失败**; 嵌套后 gate≈常数 |
| P5-04 | SCFI (LGB) | **+0.0075** (17/20 月) | GREEN | 条件创新 LGB 强 / NN 无 |
| P5-05 | RICS 几何 | ≤0.011 | RED | 短窗形态无信息; 相位破坏反演不变 |
| P5-06 | SCFI 生产验证 | +0.000849 | GREEN | 跨 learner 双确认第一步 |
| P5-07 | RealMLP spot-check | **+0.0040** / blend +0.0014 | GREEN | **SCFI 升级门禁通过**; SmallMLP 代理推翻 |

## P6/P6R 生产与检索 (2026-08-14/15)

| ID | 实验 | 结果 | 状态 | 核心结论 |
| -- | -- | -: | -- | ---- |
| P6 | RealMLP-C 生产推理 | 双窗口 +0.0014 | YELLOW | **提交候选就绪 (未提交, 等拍板)** |
| P6R-00 | 检索残差 | +0.000588 (gate 39%) | RED | 负面链 #5; KILL 检索路线 |
| P6R-01 | Local vs Global Ridge | 未跑 | SUPERSEDED | 挂起等拍板 |

## P7 幅度终裁 (2026-08-15)

| ID | 实验 | 结果 | 状态 | 核心结论 |
| -- | -- | -: | -- | ---- |
| P7-01 | 幅度门控快速版 | +0.00000 | RED | **α 单调=0; 高波动方向质量差 (0.215→0.11)**; 幅度路线整体关闭 |

---

## 方向汇总

| 方向 | 状态 | 一句话 |
| -- | -- | -- |
| 表格特征+树+MLP | ✅ 已饱和 | 0.122 平台期 → 90 特征甜点位 |
| 序列模型 (TCN/Transformer) | ❌ 已证伪 | v6 0.082 灾难; test corr 0.03 预警 |
| 幅度门控/加权 | ❌ 已证伪 | P5-03 + P7-01 双重确认 |
| 残差建模 (均值/检索) | ❌ 已证伪 | M 系列 + P4-08 + P6R 五连杀 |
| 短窗形态 (shapelet/谱) | ❌ 已证伪 | P5-05 RICS |
| 掩码预训练 / SAE latent | ❌ 已证伪 | P3-01/03 |
| 月度漂移预测 | ❌ 已证伪 | P4-05 决定性负结果 |
| 条件创新 SCFI | 🟢 **有效** | P5-04/P5-06/P5-07 三级确认, P6 候选在手 |
| 600s market 信息面 | 🟢 部分有效 | P4-01a/P5-01 (已并入 152 特征族) |
| O→T lag response | 🧪 未测试 | **当前唯一推荐** (GPT P1.5, 152 无覆盖) |

→ 已证伪方向全表: [failed-experiments.md](./failed-experiments.md)
→ 已确认结论: [research-findings.md](./research-findings.md)
→ 方法地图: [method-map.md](./method-map.md)
→ 已实验方法清单 (大白话版): [methods-tried-zh.md](./methods-tried-zh.md)
→ 提交历史: [../submissions/README.md](../submissions/README.md)
