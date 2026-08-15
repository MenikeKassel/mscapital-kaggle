# RealMLP 论文研读成果 (Better by Default, arXiv:2407.04491)

> 来源: 会话 20260815_160037_bda93b (2026-08-15 论文深度讲解 + GPT 评审核验)
> 原始材料: 本目录 (realmlp.html/txt, fig2.png, winrate_mt_reg.png, pytabkit_code/, tds_standalone_code/, html2txt.py)
> 状态: 研读完成, 迁移实验待注册 (No ID, No Experiment)
> 关联: MSCapital 的 RealMLP 复刻基线 (C-01/C-02) 与本文同源; 本文消融数字可指导配方级实验

## 1. RealMLP 到底是什么

**D 答案: 一套针对 tabular learning 系统优化过的 MLP recipe**（不是新架构/新 embedding/新 optimizer 单点发明，而是 meta-train 118 数据集上调出的组件组合 + 默认超参）。核心观点: **tabular DL 的问题很多时候不是架构不够复杂，而是训练 recipe 太差**。

## 2. 组件清单 (Vanilla MLP → RealMLP 20 步累计消融, Figure 1c + Appendix A.4)

| # | 组件 | 内容 | 消融代价 (reg/class 相对误差增加%) |
|---|---|---|---|
| 1 | Robust scale + smooth clip | median/IQR 缩放 + x/√(1+(x/3)²) 平滑截断到 (−3,3) | 去掉 +9.5% (clip, Table B.2) |
| 2 | One-hot for small cat. | ≤8 类 one-hot, 二元编码 {−1,1} | ns |
| 3 | 无 early stopping | 全 256 epochs + best-epoch 回退 | — |
| 4 | Last best epoch | 平局取最后 | +0.4% ns |
| 5 | coslog4 LR schedule | 0.5−0.5·cos(2π·log2(1+15t)), 4 cycles | vs cosine +0.4% ns; vs constant +13.5% |
| 6 | Adam β2=0.95 | 记忆 ~13.5 步 (vs 0.999 ~693 步) | **+22.8% reg** / +2.0% class |
| 7 | Output clipping (reg) | 预测裁剪到训练期观测范围 | +9.5% |
| 8 | NT parametrization | z = d^{-1/2}Wx + b | +1.3% reg / +1.1% class |
| 9 | Mish (reg) / SELU (class) | — | 激活函数间差异小 |
| 10 | Parametric activation | σ_α(x) = (1−α)x + ασ(x), α init=1, LR×0.1 | **+4.8% reg** / +0.5% ns |
| 11 | Learnable scaling layer | x' = s_i·x_i, s init=1, LR×6 (soft feature selection) | +1.0% reg / +1.4% class |
| 12 | PBLD numerical embedding | (x_i, W cos(2πw x_i + b) + b) ∈ ℝ⁴/特征 (w,b∈ℝ¹⁶) | 无嵌入 **+20.6% reg** / +2.3%; PLR +19.0%/+4.2%; PL +0.5% ns |
| 13 | Dropout p=0.15 + flat_cos 调度 | dropout/wd 按 flat_cos 调度 | constant 反而 +3.6% reg / +3.1% |
| 14 | Weight decay 0.02 (AdamW) | PyTorch 版 θ←θ−lr·wd·θ | +0.9% reg ns |
| 15 | he+5 bias init + data-driven weight init | 首前向 pass 数据依赖初始化 (行重缩放至输出方差 1) | +1.2% reg / +0.9% |

**Tier 分级** (论文证据): Tier 1 = numerical embedding (PBLD/PL)、β2=0.95、预处理 (robust+clip)、LR schedule; Tier 2 = 参数化激活、scheduled reg、scaling layer、init; Tier 3 = NTP、he+5、output clipping。**注意: 消融在 meta-train 上调过 LR, 数字不是无偏估计; β2/嵌入的 +20% 是 recipe 内敏感度, 不是普适增益。**

## 3. GPT 评审核验结论 (会话 48104, 已实测)

- **20+ 条论断逐字核验通过** (standalone 代码 mlp.py/preprocessing.py 逐行对照)
- **GPT 真新增量 5 条 (采纳)**:
  1. **target 标准化的平移敏感**: uncentered cosine 对 y 缩放不变但对**平移**敏感 → cosine 项用 raw y 方向 (不要减均值); MSE 辅助项可用标准化 y
  2. 先测 PL 再 PBLD (消融 PL≈PBLD, CI 含 0)
  3. cosine decay 先行, coslog4 押后
  4. **训练/选择指标分离**: MSE 训练 + global-cosine 选 epoch; 中途换损失会污染组件归因
  5. scaling layer 参数做跨 fold 稳定性诊断
- **纠正 4 处小错** (均不影响定性): Figure 1c 是点线图非柱状图; 均值例子 16.68→16.88; TD-S 的 Adam 非 AdamW 但 wd=0 时等价; smooth clip 梯度通道并非不重要 (hard clip 死区杀 PBLD 频率学习)

## 4. MSCapital 迁移判断 (Q1-Q5)

- **Q1 第一性原理**: 让 NN 在重尾/异质尺度/低信噪比的表格数据上"稳定吃下去"——预处理(有界输入) > 优化(β2) > 表示(嵌入) > 架构
- **Q2 三个必抄**: Robust+Clip / β2=0.95 / numerical embedding (思想, 非迷信 PBLD)
- **Q3 为什么 regression 更值**: 全部消融 reg 敏感度 >> class (nRMSE 对离群敏感)
- **Q4 MSCapital 上有效的最可能原因**: **preprocessing > optimization > 表示 > backbone** (我们的 152 特征已是强手工表示, 缺的是稳定训练配方)
- **Q5 延续方向**: 不换大模型, 继续找适合金融 tabular regression 的 recipe (scheduled reg / 周期性 LR / 目标变换)

## 5. 合并后 v2 实验顺序 (E0-E10, 待注册执行)

```text
E0  冻结 baseline (152 特征 → robust+clip → 3×256 MLP → MSE)
    固定 temporal split/seed/epochs/batch; 记录 CV cosine/MSE/best epoch
    ── 纪律: 训练用 MSE, 验证/选 checkpoint 用 GLOBAL cosine ──
E1  RobustScaler + Smooth Clip        (极低, 重尾对症, 论文 reg +9.5%)
E2  Adam β2 0.999→0.95                (一行, reg +22.8% 量级; 同步扫 wd)
E3  constant LR → cosine decay        (论文 +13.5% 的坑)
E4  Parametric Mish                   (α init=1, LR×0.1; reg +4.8%)
E5  PL 数值嵌入 (先 PL 不先 PBLD)       (验证嵌入价值; 无嵌入 reg +20.6%)
E6  PL → PBLD                         (+0.5% 量级; 152→608 维; 重扫 wd)
E7  Learnable scaling layer           (init=1, LR×6; +1.0%)
E8  dropout/wd flat_cos 调度           (constant 反而差 +3.6%/+3.1%)
E9  coslog4                           (vs cosine +0.4% ns, 优先级低)
E10 NTP + 数据驱动 init               (各 ~1%, 最后)
配方稳定后 → objective 分支 (平行三版):
  A. MSE 训练 + global-cosine 选 epoch (E0 就在用)
  B. raw-y 方向 1−cos 损失 (不要对 y 减均值!)
  C. 混合 L = MSE(y_std) + λ·(1−cos(p, y_raw)), λ∈{0.01,0.03,0.1,0.3}
  另: 去掉 output clipping (cosine 下裁剪改变方向, 只留作 A 的对照臂)
```

顺序理由: 成本低→高; 每步只动一个变量; **按"数据最缺什么"排序而非消融数字排序** (β2/嵌入并列最大但 E1 最便宜最对症重尾)。

## 6. 与现有体系的衔接

- 现有 RealMLP 复刻 (C-01/C-02) 是社区 0726 方案变体, 未含本文完整 recipe (β2=0.95/PL 嵌入/parametric activation 等)
- 本 E0-E10 系列可作为 **P8 (recipe) 阶段** 注册 (经 new_experiment.py), 与现有 152 特征管线正交
- 关键差异点: 现有生产用 cosine/混合 loss (P4-10~14), 本文建议 MSE 训练 + cosine 选择分离——与 P4-14 收口结论 (cosine 真实但小) 需对照验证
- 待用户拍板: loss 策略 / E5 起点 (PL vs PBLD) / 是否开跑
