# GPT 量化路线评审核验报告 (2026-08-15)

> 评审来源: GPT 第三轮 (7 条新路线 + 2 条不推荐)
> 核验方法: 三件套重叠性核对 (method-map/registry/failed-experiments) + arXiv 引用实证 + 源码检查
> 状态: **讨论定稿阶段, 等用户拍板后执行**

---

## 1. 逐条核验表

| # | 路线 | 重叠性核对 (三件套) | 论文引用实证 | 可行性 | 裁决 |
|---|---|---|---|---|---|
| ① | **Month-DG (V-REx/GroupDRO)** | ✅ **真空白**: DRO/IRM/V-REx/domain/invariant 零命中; P0-02 AV=检测、P4-07 漂移预测(RED)、E-03 审计均非"训练约束"; P3-02 状态条件化是市场状态非月份; hard-regime/MoE(🧪)是专家混合非方差惩罚 | ✅ 1911.08731 GroupDRO 真实 | ✅ label 有 month 列; 同 C 系列骨架加 loss 方差项 | **S 级成立** |
| ② | **TabM/BatchEnsemble** | ⚠️ **部分重叠**: 项目 RealMLP 复刻 (src/mscapital/models/realmlp.py) **已有 n_ens=16 联合训练** (ScalingLayer/Embedding 参数带 n_ens 维, 与 lb142 v10 RealMLPEns 同构); RESULTS L174 "RealMLP_RQ n_ens=16 完整复现" | ✅ 2410.24210 TabM 真实 | 已有类似物 | **降级 S→B**: 泛化"联合 ensemble"已做过; TabM 增量仅为 mini-block 参数共享变体 |
| ③ | **NCL diversity learning** | ✅ **真空白**: residual 五连杀 (P4-08/P5-03/P6R-00/M-01/F011) 全是"残差目标"; NCL 是"误差相关性惩罚", 零命中; P4-10 (INVALID) 是目标层非多样性层 | ✅ 1803.00314 (Brown, NCL 经典) 真实 | ✅ 需要 v7 OOF 预测 p; L = L(B,y) + λ·corr(e_B, e_p) | **S 级成立** |
| ④ | **监督对比回归** | ✅ **真空白**: P3-01/03/05 全为无监督 (SAE/掩码/NHP); 用 y 组织 latent 零命中 | ⚠️ **引用名错**: 2210.01189 实际是 **Rank-N-Contrast (RNC, ICLR 2023)** 而非 "SupCR"; 方向 (监督对比回归) 正确, 引用名需修正 | ✅ 可实现 (encoder + 对比 loss) | **A 级** (排在①③后) |
| ⑤ | **预测中性化** | ✅ **真空白**: P7-01/P5-03 是 gate/调制 (RED); neutralization 是减法剥离 p−β^TZ, 零命中 | 无引用 (通用技术) | ✅ 最便宜, 15-30 分钟, 无需重训 | **A- 级, 立即测** |
| ⑥ | **GP alpha mining** | ✅ 真空白 (人工 FE 已到顶 ≠ 自动搜索) | ✅ 2412.00896 真实 | ✅ 但过拟合风险大 (1.26M×152) | B 级 (GPT 自评正确) |
| ⑦ | MDN/quantile | — | 2510.25001 未实证 (低优先无需) | 幅度线已关闭 | 不推荐 ✓ |
| — | LTR (不推荐) | P4-15 身份取证 RED → 横截面结构不可恢复 | ✅ 2012.07149 真实 | 结构性不可用 | 同意不推荐 ✓ |

## 2. 关键修正 (2 处)

1. **② TabM 降级**: 项目 n_ens=16 (realmlp.py L61/L415/L429) 已是"单网络内多成员联合训练"——TabM 的"一个网络多个成员"思想**已落地**; 真正没做的只是"mini-block 参数共享"这一工程变体, 收益预期弱于"换一条没走过的训练范式"。
2. **④ 引用修正**: 2210.01189 = Rank-N-Contrast (RNC), 不是 SupCR。方向不变 (监督对比回归), 但按 RNC 的实现走 (对比 loss 用 rank-based distance)。

## 3. 合并后的路线图 (GPT 三轴 + 现有 C 系列)

```text
                    现有 Alpha (152/SCFI/market/lb142)
                              │
      ┌───────────────┬───────┴────────┬───────────────┐
      ▼               ▼                ▼               ▼
 Optimization     Robustness       Diversity        Cheap probes
      │               │                │               │
 C 系列 (拆解)     P9-DG (Month)     P9-NC (NCL)    ⑤ neutralization
 C-05~C-15        λ∈{0,.1,.3,1}     λ·corr(e_B,e_v7)  (30min)
 E0-E10 进行中     ↓ G0 门禁          ↓ G0 门禁          ↓
      └───────────────┴────────────────┴───────────────┘
                              ▼
                       Final Ensemble
```

- **P9-DG**: 同 C 系列骨架 (152→robust+clip→3×256 MLP→MSE), loss = mean(L_m) + λ·var(L_m), λ∈{0,0.1,0.3,1.0}; month 分组来自 label.month; PSEUDO 33-70 同协议。**失败判据预注册**: 无 λ 单调改善 + 月度正比例不升 → RED。
- **P9-NC**: 需要 v7 OOF 预测 (项目已有 frozen OOF), 惩罚 corr(e_B, e_v7), λ∈{0,0.1,0.3,1.0}。**注意与 residual 区分**: B 直接预测 y, 不拟合残差。
- **⑤ neutralization**: 直接在 PSEUDO 预测上 p' = p − γ·ŷ_Z, Z = {mid_std, spread, tx activity, order activity, predicted|y|}, γ∈{0,.25,.5,.75,1}, calibration 选, frozen 审判。**P7-01 的机制假设直接复用** (高波动方向质量差)。

## 4. 与 27 个 RED 的边界声明 (防误判)

- residual 五连杀 (F011) 判死的是"拟合残差目标"; NCL 是"惩罚误差相关性"——**不同目标函数, 不重叠** ✓
- P7-01/P5-03 判死的是"幅度调制/gate"; neutralization 是"预测空间减法"——**不同操作, 不重叠** ✓
- P3 系列判死的是"无监督 latent"; RNC 是"监督对比"——**不同监督信号, 不重叠** ✓
- P4-07 (漂移不可预测) 判死的"预测漂移"; P9-DG 是"惩罚跨月方差"——**不同目标, 不重叠** ✓ (但 P4-07 提示: DG 收益可能有限, 预期管理)

## 5. 执行优先级 (等拍板)

1. **⑤ neutralization** (30 分钟, 零训练成本) — 先测, 顺手做
2. **P9-DG** (半天, 4 个 λ) — GPT 首选, 真空白
3. **P9-NC** (半天) — 需要 v7 OOF, 与 C 系列并行
4. RNC probe (A 级) — 排在 1-3 后
5. TabM mini-block (降级 B) — C 系列出结果后作为 n_ens 变体评估
6. GP alpha mining (B) — 默认不做, 除非 1-5 全败
