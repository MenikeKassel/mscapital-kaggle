# Hypothesis-Verification Loop 台账 (2026-08-15 起)

> 工作流: 每个提升想法 = 假设 → 预注册验证实验 (C-05 同骨架, PSEUDO 33-70, fold-local 预处理, best-epoch cosine 选择) → 判定 (GREEN/RED/YELLOW) → 通过者叠加进候选链
> **提交门禁 (用户 2026-08-15 拍板): 本地 PSEUDO > 0.16 才允许提交 Kaggle**
> 当前本地最高: 四折均值 0.1435 (单折 0.1569) / v7 PSEUDO 0.13968 / C-05 E0 基线 0.113261
> 缺口: 0.1435 → 0.16 = **+0.0165** (需多条假设叠加或 1-2 个大增益)

---

## 1. 假设池 (按验证状态)

### 已验证 (GREEN, 可叠加)
| 假设 | 增益 (PSEUDO) | 验证 | 状态 |
|---|---|---|---|
| H0 **SCFI 兑现 (152+73Z)** | plain +0.0070 / **RQ 生产 +0.0044** (30/38 月) | P9-SCFI-REALIZE + P9-SCFI-RQ | ✅ **生产验证 GREEN** |
| H0b **Z_O2 二阶条件 (152+41Z)** | **+0.0060** (28/38 月) | P10-FM order | ✅ GREEN (plain, RQ 待验) |
| H1 预测中性化 (nuisance 剥离) | **+0.0014** (frozen 51-70) | P9-NEUT | 🟡→ 待叠加验证 |
| H2 robust+clip 预处理 | +0.0011 | C-06 | ✅ 已进基线 |
| H3 Adam β2=0.95 | +0.0009 | C-07 | ✅ 已进基线 |
| H4 Month-DG (V-REx, λ=3) | +0.0003 | P9-DG | 🟡 边缘, 不推荐叠加 |
| H13 M1 L2 档位 (20 特征) | +0.0008 | P10-FM market | 🟡 边缘 (未达门禁) |

### 已证伪 (RED, 不重复)
- O→T 跨表时序 (P8-01A, 0/71月), NCL 误差互补 (P9-NC, err_corr≈0.99), 残差五连杀, gate/幅度调制 (P7-01), 无监督表示 (P3 系列), 短窗形态 (P5-05)…
- **M2 事件/跳跃结构 (P10-FM market, −0.0002)**
- **TX H1/H2/H3 分段聚合+大单拆分 (P10-FM transaction, −0.0032; 22/38 月正但整体负)** — 与 M-01 一致: tx 聚合形态已到顶

### 待验证 (按预期价值 × 成本排序)
| # | 假设 | 依据 | 成本 | 状态 |
|---|---|---|---|---|
| H5 | C 系列 E3-E10: cosine decay → Parametric Mish → PL → PBLD → scaling layer → scheduled reg → coslog4 → NTP/init | 论文 20 步消融 (Tier 1-2); 每个预期 +0.0005~0.0015 | 8×12min GPU | ⏳ C-08~C-15 已注册 |
| H6 | neutralization 叠加到生产融合 (v7/v8b OOF) | H1 在 C-05 上 +0.0014; 零成本后处理 | 30min | 🔄 本轮 |
| H7 | 5-model refit ensemble (论文 B.3: reg 8.7% error reduction) | 论文集成增益 > 任何单组件且正交 | 5×训练 | 🧪 |
| H8 | cosine objective (λ_cos 混合, v10 实战 λ=0.05) | lb142 v10 实证; P4-11~14 审计中 | 训练变体 | 🧪 |
| H9 | 乘法双头 pred = sign×mag (market 幅度 0.43 兑现) | P5-02I 建模含义 | 半天 | 🧪 |
| H10 | n_ens 变体 (TabM mini-block 参数共享) | 已有 n_ens=16, 增量存疑 | 1-2天 | 🧪 低优先 |
| H11 | 局部回归终裁 (P6R-01) | 挂起中 | — | 🧪 |
| H12 | GP alpha mining | 过拟合风险大 | — | 默认不做 |

## 2. 循环状态机

```text
新假设 → 预注册 (骨架/切分/门禁写明) → 跑 PSEUDO → Δ > +0.001 且方向稳 → GREEN 进候选链
                                              → |Δ| ≤ 0.001 → YELLOW (边缘, 不叠加)
                                              → Δ < 0 → RED (登记 failed-experiments)
候选链叠加验证 → 本地最高分 > 0.16 ? → 是 → 提交 (跑 calibration 门禁: 双窗口一致+分布/尺度)
                                    → 否 → 继续下一轮假设
```

## 3. 当前执行轮次

**Round 1 (进行中)**: H6 neutralization 叠加验证 (生产级 OOF) + H5 第一批 (C-08 cosine decay, C-09 Parametric Mish)
**Round 2 (计划)**: H5 剩余 + H7 refit ensemble
**Round 3 (计划)**: H8 cosine objective + H9 乘法双头
