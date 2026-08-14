# MSCapital P5-C — RICS Deterministic Cross-Channel Probe 报告 (2026-08-14)

> 实验: `scripts/p5c_rics.py` | 产物: `output/p5c_rics/{results.json, p5c_preds.npz, p5c_layer_preds.npz}`
> 任务书: "P5-C RICS Deterministic Cross-Channel Probe" (2026-08-14, §6)
> 判定: **KILL** (corr(R4,y)=−0.006 < 0.09; phase ≤ covariance; late 无增益; R4 反转相关为负)

## 12.1 Hypothesis

Market alpha 主要来自 short-range (≤30s / last-10 网格步) cross-channel geometry
(协方差/对称 lag/相位相干), 而非全窗口序列动力学。**—— 否, 实验证伪。**
有效信息在 200 步全窗口 (M0 corr 0.086), 不在 last-10 的确定性几何统计 (≤0.011)。

## 12.2 Data

| 项 | 值 |
|---|---|
| 输入 | market.feather → 200 步均匀 3s 网格, 取**最后 10 步** (~30s) × **12 核心通道** (L1/L2 四价四量 + avgprice + mid + spread1 + depth1; 经济含义选择, 非暴力搜索) |
| 特征阶梯 | R0 flatten 120d → R1 +moments 96d → R2 +Cov/Corr 132d → R3 +对称/奇 lag (k=1,2,3) 396d → R4 +相位 (Re/Im/cosΔφ/\|sinΔφ\|, f=1,2) 552d |
| 下游 | SmallMLP [256×2] + cosine loss, 15ep, AdamW 1e-3, seed 42 |
| 协议 | P5-01 复刻: train 21-40, alpha 41-50 选 (blend canonical baseline), frozen 51-70 |
| 参照 | **M0-ref = P5-01 200 步 Conv1D 原版同协议重训** (当前 market 模型, corr≈0.086) |
| 时间反演 | R4 + M0 同一 frozen 模型: pred_fwd vs pred_rev (输入反转), even/odd 分解 |

## 12.3 Leakage audit

- [x] 特征全部确定性变换 (无训练统计, 除 MLP 标准化 stats 只来自 train 21-40)
- [x] alpha 只在 41-50 选; 51-70 完全冻结
- [x] R4 反转分解用 21-40 训练模型对 51-70 推断 (无 frozen 期拟合; even/odd blend 网格
      在 frozen 段调权 — 明确标注为诊断, 不参与判定)
- [x] M0-ref 与 P5-01 同架构/同 seed/同协议

## 12.4 Baselines

| 基线 | corr_y (51-70) | Δfrozen |
|---|---:|---:|
| M0-ref (200 步 Conv1D, 本实验重训) | **+0.0861** (P5-01 报告 0.0861, 完美复现) | −0.000031 |
| canonical baseline OOF (blend 锚) | — | — |

注: P5-01 的 frozen Δ (+0.0009) 是 blend v7_like 锚; 本实验统一 blend canonical 锚,
Δ 数值不可直接比 (锚更强), 但 R0-R4 与 M0 的**同锚比较**有效。

## 12.5 Results (51-70 frozen, 同锚)

| Model | dims | corr_y | Δfrozen | late (61-70) | months>0 |
|---|---:|---:|---:|---:|---:|
| R0 flatten | 120 | +0.0111 | −0.000070 | +0.000011 | 8/20 |
| R1 +moments | 216 | +0.0018 | −0.000166 | −0.000201 | 7/20 |
| R2 +cov/corr | 348 | +0.0048 | +0.000000 | +0.000000 | 0/20 |
| R3 +lag even/odd | 744 | −0.0002 | +0.000000 | +0.000000 | 0/20 |
| R4 +phase | 1320 | **−0.0060** | +0.000000 | +0.000000 | 0/20 |
| **M0-ref** | 3600 | **+0.0861** | −0.000031 | — | — |

## 12.6 Reversal decomposition (§6.5-6.6, 同一 frozen 模型)

| 模型 | corr(fwd,rev) | MAE(fwd−rev) | cos_even(y) | cos_odd(y) |
|---|---:|---:|---:|---:|
| R4 (相位) | **−0.6906** | 0.395 | −0.0066 | −0.0005 |
| M0 (200 步) | **+0.0688** | 0.390 | +0.0562 | +0.0558 |

- R4 对输入反转**高度敏感且符号翻转** — 相位特征 (复值) 破坏反演不变性, 与
  P5-02I 的"无时间箭头"实锤直接冲突 → 该表示方向被架构内证据否决
- M0 近反演不变 (corr 0.069) 且 even/odd 分解各自携带等量信号 — 反演不变性
  存在于**全窗口学习表示**, 不在 last-10 确定性相位统计

## 12.7 阶梯解读 (增益从哪一步出现 — 答案是: 都不出现)

R0→R1 变差 (统计量抹掉形态), R2 协方差略回升但 Δ=0, R3 lag 无信号, R4 相位
负值且过拟合 (loss 0.90→0.39, corr −0.006)。**没有任何一步出现增益**:
10 步窗内的确定性几何 (无论时域统计、二阶协方差、对称 lag 还是频域相位)
都不携带 market alpha。R4 训练 loss 大幅下降而验证 corr 为负 = 高维相位特征
在 35 万样本上过拟合噪声形态。

## 12.8 Failure analysis

最可能类别: **Hypothesis false (信息不在 last-10 确定性几何)** + 部分
**Representation failed** (R4 相位特征形式错误 — 未做 Welch 平均/多窗, 单窗
periodogram 相位噪声大, 任务书 §6.4 已预警; 但 R0-R3 的失败与表示形式无关,
是窗口本身的问题)。

机制证据 (P5-02I 重新解读): block10 shuffle 崩 69-81% 但 last-10 单独 ≈ 无信号 →
**"≤10 步形态"不是"最后 10 步"; alpha = 同一短形态在全窗口多位置的重复出现,
且需要全窗口上下文**。任何 last-10-only 架构都不可能有 M0 级表现。
R4 反转相关 −0.69 进一步说明: 相位特征构造本身违背反演不变, 该表示线关闭。

## 12.9 Decision

**KILL** (任务书 §6.9):
- corr(R4,y) = −0.0060 < 0.09 ✓杀
- phase ≤ covariance (R4 corr −0.006 < R2 0.005) → 相位无独立价值 ✓杀
- 唯一通过项是 late≥0 (因 Δ=0) — 无意义
- 按 §6.9: **wavelet / shapelet / spectral CNN / large phase network 方向全部关闭**

## 遗留/备注

- p5c_layer_preds.npz 含 R0-R4 + M0 的 sel/fr 预测 (可复用, 无需重训)
- R4 的相位特征实现是单窗 rFFT (无 Welch 平均) — 任务书 §6.4 明确首轮 probe
  不需要多窗; 本结果同时意味着"升级版"不再需要 (窗口层面已死)
- 本实验未测 TRIS random-shapelet 臂 (deepseek 提案): R0-R4 已证 last-10 窗口
  无信息, shapelet 在同一窗口上不可能翻身 → 按 §6.9 与任务书 §7 一并关闭
