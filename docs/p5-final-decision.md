# MSCapital P5 Final Decision — 三 Probe 汇总裁决 (2026-08-14)

> 任务书: "下一阶段严格实验执行任务书" Step 6 (2026-08-14)
> 三 Probe 报告: `docs/p5a-mag-gate-report.md` / `docs/p5b-scfi-report.md` / `docs/p5c-rics-report.md`
> 全部判定基于 strict temporal validation + nested OOF + monthly stability + orthogonality (任务书 §8), **未提交任何 Kaggle**

---

## 1. 三个假设的裁决 (以真实实验回答, 非理论偏好)

### Q1: MAG 是否成立? → **KILL** (P5-A)

| 证据 | 值 |
|---|---|
| 嵌套 Δcosine (51-70) | **−0.000146** (非嵌套 +0.000138 → 嵌套后消失) |
| gate 权重 | a≈1, std=0.011 (优化器找不到条件幅度形状) |
| 置换 m̂ 对照 | +0.000004 ≈ 0 |
| 月度 | 6/20 正 |

market 幅度预测真实存在 (corr 0.43-0.47, P5-02I 复现), 但**无法通过逐样本幅度加权
(v7 × a_bin(m̂)) 变现为 cosine 增益** — cosine 全局尺度不变, v7 的 |pred| 形状已是
当前可学最优。**MAG-MoE / 幅度调制完整版 (COC) 关闭。**

### Q2: SCFI 是否成立? → **CONDITIONAL CONTINUE** (P5-B + learner spot-check)

| 证据 (LGB, 同 learner 对照) | 值 |
|---|---|
| ΔC (152+Innovation) | **+0.0075** (B1 +0.0059 / late +0.0091) |
| Δ(D−E) (超出 capacity control) | **+0.0016** (双块为正) |
| 逐月 C-vs-A | **17/20 月正** |
| blendΔ (vs canonical, holdout 调权) | +0.00094 |
| corr(C, canon) | 0.872 (>0.80, LIVE 未达) |
| corr(C, B) | 0.959 (贴 0.95 kill 条款; 直接 Δ 证据反驳等价, §9 标 caveat) |

| NN spot-check (SmallMLP×3 seeds, 同协议) | 值 |
|---|---|
| ΔC | +0.0011 / −0.0009 → avg **+0.0001** (12/20 月) |
| NN-C corrCanon / blendΔ | 0.649-0.691 / +0.00096…+0.00145 |

**结论**: Innovation 特征 (Z = 事件流 − E[事件流|市场状态], MAD 稳健尺度) 对
**LGB 家族是强增益且稳定** (+0.0075, 17/20 月), 对 **SmallMLP 家族无增益** —
learner 依赖。加上原始 side×action 聚合特征 (73 个, 独立于 innovation, Arm B
+0.0058) 本身即新信息面。**条件强度/点过程/FiLM 升级阶梯维持门禁 (未达 LIVE/STRONG)。**

### Q3: RICS 是否成立? → **KILL** (P5-C)

| 层 | corr_y | 判读 |
|---|---:|---|
| R0 last-10 flatten | +0.0111 | 10 步窗几乎无信号 |
| R1 +moments | +0.0018 | 统计量抹掉形态 |
| R2 +cov/corr | +0.0048 | 二阶结构无独立 alpha |
| R3 +lag even/odd | −0.0002 | 无 |
| R4 +phase (1320d) | **−0.0060** | 过拟合, 反转相关 **−0.69** (破坏反演不变) |
| M0-ref (200 步, 同协议重训) | **+0.0861** (P5-01 完美复现) | 参照有效 |

**wavelet / shapelet / spectral CNN / large phase network 全部关闭**。
机制: P5-02I 的"≤10 步形态" ≠ "最后 10 步" — alpha 是全窗口多处重复的短形态
+ 上下文, last-10 确定性几何 (无论统计/协方差/lag/相位形式) 都不携带。

---

## 2. 0.142 之后最值得投入的主线

**短期 (低成本, 直接可做):**
1. **152 + 73 raw O/T 聚合 (side×action 拆分/burstiness/size 分位) 重训 LGB/CatBoost
   表格模型, 进入 blend** — Arm B LGB 证据 +0.0058 (B1/B2 一致), 特征已落盘
   (`output/p5b_scfi/raw_ot_agg.parquet`), 半天内可出生产级验证
2. **152 + Z innovation 重训 LGB 进入 blend** — Arm C +0.0075, blendΔ +0.0009;
   与 canonical corr 0.87 → 有 blend 价值 (PSEUDO 门禁 + 提交纪律照旧)

**中期 (门禁后):**
3. **RealMLP 精确 spot-check** (152+Z vs 152, 生产 learner): 决定 SCFI 是否升级 —
   若 RealMLP 也受益 → conditional intensity 阶梯 (任务书 §5.14) 才值得开
4. PSEUDO 门禁通过后: 生产推理 (全量重训 + test) + 提交校准 (§7e 纪律)

## 3. 应彻底关闭的路线

- MAG-MoE / 幅度调制 (4-bin gate 已证伪, 更复杂架构期望更低)
- 条件创新的神经网络条件模型 / Hawkes / FiLM (LGB 有增益但 NN 无 — 表示层升级无依据)
- RICS 全线: wavelet / shapelet / spectral CNN / phase-aware encoder (last-10 窗口无信息)
- 长序列 Transformer / TCN (继承停止清单 + P5-02I)

## 4. 下一次真正值得运行的大模型实验

**没有"下一次大模型实验"。** 三 Probe 的诚实结论: 增量来自**特征表示层**
(条件化 surprise + side×action 聚合), 不是模型架构。下一步是
特征 × 生产 learner (RealMLP/表格) 的组合验证 + 融合, 而非新模型。
若 RealMLP spot-check 通过, 再谈 conditional intensity; 否则 SCFI 止步于表格族,
资源转向融合工程。

## 5. 遗留与纪律备注

- P6R-01 终裁实验 (Local vs Global Ridge) 仍挂起 — 与本次结论无关, 由用户另行拍板
- 全部产物: `output/p5a_mag_gate/`, `output/p5b_scfi/`, `output/p5c_rics/`
  (预测/指标/特征全部落盘, 可复用)
- 三个假设的否定 (MAG/RICS) 与部分肯定 (SCFI-LGB) 均为预注册判据下的真实结果;
  无 threshold hunting, 无 protocol 修改, 无 LB 驱动 (§9/§16 遵守)
