# 实验总览（截至 2026-08-12）

## 1. 当前状态

- 当前最好 Public LB：**0.142**（v8b）
- 当时排名：**#30/107**
- 自研最好体系：**v7，Public LB 0.135**
- 当前纪律：冻结提交；候选必须先通过 PSEUDO、时序稳定性、预测尺度和相关性门禁
- 目标：前 10 约为 0.151，当前差距约 0.009

## 2. 评价与验证协议

比赛指标为未中心化 cosine similarity：

\[
\operatorname{cos}(\hat y,y)=
\frac{\hat y^\top y}{\lVert\hat y\rVert_2\lVert y\rVert_2}
\]

数据存在显著月份漂移，因此实验按模型族分别校准：

- **Regime A：传统表格模型。** 历史上 Public LB 通常比 PSEUDO 低约 `0.008–0.010`。
- **Regime B：新表示/RealMLP。** 当前只有 v7 一个独立定标点，差值为 `0.004683`，不能外推为通用公式。
- **Regime C：外部冻结预测。** 没有共同 validation，只能作为 ensemble probe，不能进入本地分数校准。

任何新候选必须同时报告：

1. 多个 temporal fold 和 PSEUDO 分数；
2. valid/test 的 mean、std、分位数；
3. `std_test/std_valid`；
4. 与 v5、v7、RealMLP 和当前最好提交的预测相关性；
5. 融合前后的尺度与均值变化。

## 3. 成绩阶梯与归因

| 阶段 | 版本 | 方法 | PSEUDO | Public LB | 相对前一阶段 |
|---|---|---|---:|---:|---:|
| 基线 | v1–v3 | 官方特征 + 树模型融合 | 0.129–0.130 | 0.122 | — |
| 表示归一化 | v4 | R2 归一化 + temporal 权重 | 0.131316 | 0.123 | LB `+0.001` |
| 微观结构 | v5 | v4 + 22 个无量纲 primitive | 0.134871 | 0.125 | LB `+0.002` |
| 序列模型 | v6 | v5 + TCN（7%） | 0.137920 | 0.082 | LB `-0.043`，失败 |
| 事件动力学 | v7 | v5 + 152 特征 RealMLP（20%） | 0.139683 | 0.135 | LB `+0.010` |
| 外部正交表示 | v8a | v7 + LB142 reference（30%） | N/A | 0.139 | LB `+0.004` |
| 外部正交表示 | v8b | v7 + LB142 reference（50%） | N/A | **0.142** | LB `+0.007` |

三次有效台阶：

```text
0.122 ──R2 + micro──> 0.125
0.125 ──event dynamics + RealMLP──> 0.135
0.135 ──external orthogonal alpha──> 0.142
```

## 4. 已完成实验

### 4.1 基线、特征数与模型融合

- 官方 90–92 特征附近是稳定甜点位。
- 继续增加到 90 个以上曾出现 CV 上升、LB 下降。
- 树模型、MLP 和时间权重的主要价值来自融合，而不是单模型极限调参。

### 4.2 Temporal Matrix 与漂移诊断

- Adversarial validation 发现盘口相关特征漂移最大。
- 原始 MLP temporal matrix 使用 last epoch，曾对 MLP 不公平；恢复 best state 后显著改善。
- R2 归一化在多个 temporal fold 为正，说明无量纲表示延长了 alpha 寿命。

### 4.3 微观结构特征

首轮 22 个 primitive 包括：

- add/cancel imbalance、归一化 OFI、事件到达率和 burstiness；
- 成交强度、大单占比和 fast–slow flow；
- microprice gap、相对价差和 L2 深度不平衡。

结果：PSEUDO 从 `0.131316` 提升到 `0.134871`，Public LB 从 `0.123` 提升到 `0.125`。

### 4.4 TCN 序列模型

- PSEUDO 上表格 + TCN 融合约提升 `0.0044`。
- test 上 TCN 与表格预测相关性降至 `0.03`，而验证期约为 `0.30–0.40`。
- v6 Public LB 只有 `0.082`，确认序列模型发生严重分布外退化。

结论：冻结 TCN/Transformer 架构搜索；以后低相关性必须和相关结构、尺度迁移一起判断。

### 4.5 事件动力学与 RealMLP

- 152 个事件动力学特征与 RealMLP_RQ 形成新的信息表示。
- RealMLP 单模型 PSEUDO：`0.138560`。
- v5 表格模型 PSEUDO：`0.134871`。
- 实际 v7 融合 `0.8 × table + 0.2 × RealMLP`：`0.139683`。
- v7 Public LB：`0.135`，Regime B 差值：`0.004683`。
- table/RealMLP Pearson：valid `0.8658`，test `0.8263`。
- v7 `std_test/std_valid = 0.7447`，存在明确幅度迁移。

本地原始尺度的最优 RealMLP 权重约为 0.03，但它只用于 forensic 诊断，不用于提交或继续做 Public LB 权重扫描。

### 4.6 LB142 reference 融合

- v7 与 reference 的 test 相关性约 `0.823`。
- 30% reference 得到 Public LB `0.139`。
- 50% reference 得到 Public LB `0.142`。
- 精简包的冻结融合公式为：

```text
ens5 = mean(unit(v9_big, v9_ctrl, v9_deep, v9_v3grid, v9_v3grid_big))
prediction = 0.6 * ens5 + 0.4 * unit(v10)
```

精简包不包含训练使用的 `grids/` 和 `factors/`，因此目前只能验证冻结预测，不能完整复现训练数据表示。

## 5. 负面结果库

| ID | 实验 | 结果 | 形成的规则 |
|---|---|---|---|
| N001 | 自定义 cosine loss | CV 上升、LB 下降 | 标准 loss 训练，cosine 只用于评价 |
| N002 | 大量追加特征 | CV `0.1409`、LB `0.116` | 特征按族小步 ablation |
| N003 | 深度 Transformer | CV `0.1549`、LB `0.120` | 不以架构复杂度解决漂移 |
| N005 | v5 + TCN | PSEUDO 正增益、LB `0.082` | test 相关结构和尺度是硬门禁 |
| N006 | 第二轮简单相对化 | PSEUDO `-0.0012` | 不做全量 rank/relative 变换 |
| N007 | RealMLP PSEUDO v10/v11 | 重启与产物口径错误 | 环境幂等、恢复 best EMA、产物顺序回归测试 |

## 6. 当前候选与优先级

### P0：LB142 信息归因

目标不是再做一次融合，而是回答 v9 grids 与 v10 factors 分别覆盖了哪些信息，并建立成员相关矩阵和信息缺口矩阵。

### P1：Residualized Dynamics V2

使用 v7 的严格 OOF 预测构造残差目标：

\[
r=y-\beta\hat y_{v7}
\]

新模型只学习现有体系未覆盖的部分。首批特征按族独立验证：

- 多尺度 `diff(1/2/4/8)`、加速度和反转；
- EWMA `3/8/16` fast–slow；
- `change/dt`、事件强度加速和 recent/old intensity；
- slope、excursion、reversal count、sign persistence；
- 盘口冲击后的深度恢复与 microprice 偏离衰减。

### P2：Residual Market-State KNN

用波动、价差、深度不平衡、OFI、成交方向和事件强度构造低维状态，只允许从历史月份检索邻居，预测 v7 残差。

### P3：低阶 Path Signature

在 midprice return、spread、depth imbalance、OFI、trade imbalance 和 event clock 上计算短/中/长窗口二阶 signature，控制维度后交给树模型或残差模型。

## 7. 新实验晋级门槛

建议同时满足：

- 多个 temporal fold 均为正，不依赖单一近期月份；
- 与当前体系融合后的 PSEUDO 增量达到约 `+0.0015`；
- valid/test 的预测相关结构没有异常坍缩；
- 预测尺度和均值没有突变；
- 相对 v7/LB142 提供可解释的独立信息；
- 通过门禁后才生成提交候选，不使用 Public LB 连续扫权重。

更详细的实验记录见仓库根目录的 `RESULTS.md`；校准制度见 `docs/calibration.md`。
