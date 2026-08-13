# MSCapital 下一代方法决策报告

> 日期: 2026-08-14 | 状态: 调研完成 (3 子代理部分失败, 主代理补齐关键缺口)
> 目标: 找到让自研 LB 从 0.135 跨到真正 0.14x 的下一代方法
> 原则: 信息表示 > 学习机制 > 模型架构 > 超参数 (但以实战证据为准)

---

# 1. Executive Summary

## 当前最值得实验的 Top 5 新方法

| # | 方法 | 核心洞见 | 实战证据 | 实现成本 | 互补性 |
|---|---|---|---|---|---|
| 1 | **监督自编码器 (SAE) latent 特征** | 让模型学习 32-64d 市场 latent, 不再人工造特征 | **两个独立冠军**: JS 2021 1st (单模型 private 6022.202 即夺冠) + DRW 2025 1st (AE 合成 8 特征, 作者称"很重要") | 低 (1-2天) | 高 (新隐变量, 非人工特征) |
| 2 | **市场状态条件化 (Reconditioning) 升级** | 残差可预测性随市场状态变化 → 按状态条件化建模 | **自研 E02 gate=True** (残差 cos +0.0132, 95% CI 不跨 0) | 低 (1-2天) | 高 (直接针对漂移) |
| 3 | **TinyLOBERT 掩码事件预训练** | 消息级 tokenization + masked 重建 → 事件级 latent | LOBERT (arXiv:2511.12563, 2025-11) 论文自报; 无比赛证据 | 中 (3-5天, 云端 GPU) | 高 (事件级 vs 窗口级表示, 完全不同层) |
| 4 | **OrderFusion 式 2.5D 时空网格** | 价格档 × 时间 × 深度 的网格编码, 非 summary | OrderFusion (arXiv:2502.06830, 电力市场订单簿) 论文自报 | 中 (3-5天) | 中 (与 M02 部分重叠, 需离散价格档) |
| 5 | **Neural Hawkes 事件强度特征** | 预测 λ(buy_add/sell_add/cancel/trade) + 下一事件时间, 作为 hazard 特征 | Neural Hawkes (1612.09328); LOB 应用 (2502.17417, 做市模拟) 论文自报 | 高 (5-7天, 事件级 MLE) | 中 (可能被 M01-A 事件率覆盖) |

## 如果今天只能选两个开始写代码

**P3-01 (SAE latent) 和 P3-02 (状态条件化)**。

- **P3-01**: 两个独立 Kaggle 冠军 (JS 2021、DRW 2025) 都靠 AE 类方法赢, 是全部候选中实战证据最强的; 防泄漏做法 (CV split 内联合训练) 与项目 canonical OOF 基础设施完美契合; 1-2 天可出结果。
- **P3-02**: 项目 8 个 M/E 系列实验里唯一 gate=True 的方向, 直接对准"漂移"这一核心痛点; 成本最低, 且与 P3-01 互不冲突 (一个改表示, 一个改建模机制)。

---

# 2. Similar Competition Matrix

| 比赛 | 年份 | 与 MSCapital 相似度 | Top solution | 核心方法 | 公开效果 | 可复现度 | 最值得迁移内容 |
|---|---|:---:|---|---|---|:---:|---|
| DRW Crypto Market Prediction | 2025 | S+ (未来数据评分, 时间漂移即赛题; 匿名特征) | 1st: A_A (Tony271YnoT) | 3层MLP (SGD, 0.6MSE+0.4Pearson) + XGB; 特征聚类medoid 780→60 + SHAP筛选; **AE合成8特征**; purged group TS CV | Private 0.13959 (1st) | L2 (writeup+讨论, 代码部分公开) | AE 合成特征; 混合loss; SGD; 特征去冗余 |
| Jane Street Real-Time Market Data | 2024 | S+ (实时评分, 匿名特征, 漂移) | 58th: I2nfinit3y 队 (公开最高 writeup) | **TabM** (3层512d×16成员, R2 loss) + **AutoencoderMLP** (线性96d编解码 + 5层MLP) + **online learning** (每日回标重训5epoch) + GBDT/Ridge | Private 0.0092 (58th); TabM 单模 lb 0.0077 | L2 | R2 loss; time_id 类别化; online learning 机制 (MSC N/A: 无实时回流) |
| Jane Street Market Prediction | 2021 | S (匿名特征, 未来真实数据) | 1st: VECTOR/Yirun Zhang | **Supervised Autoencoder + MLP**: AE 重建 + 监督 target 进编码器, 联合训练防泄漏; Gaussian noise 增强; swish; 5-fold 31-gap purged; 3 seed; 只用最后2折模型 | **单模型 private 6022.202 = 1st** | **L3 (notebook 全公开)** | SAE 结构 + 防泄漏联合训练 (直接可搬) |
| Ubiquant Market Prediction | 2022 | S (匿名特征, Pearson) | 2nd: Stenner | Robust CV + LGBM (5 模型, CV corr early stopping) | 2nd | L2 | 已部分吸收 (canonical OOF 制度) |
| Optiver Trading at the Close | 2023 | S (盘口特征, 价格方向) | 1st: hyd | 特征工程 + GBDT/NN + weighted post-processing (5.4457→5.4405) | 1st | L1 | 高阶交互已吸收 (M05); zero-sum 需目标结构证明 |
| Optiver Realized Volatility | 2022 | S (匿名特征) | 1st: Nearest Neighbors | KNN 历史状态 + LGBM/1D-CNN/MLP 融合 | 1st | L2 | KNN 已试 (M05 gate False); 不重复 |
| G-Research Crypto | 2022 | A (多资产时序) | top: AE/encoder + MLP (细节未核实*) | latent representation | — | — | 待核实 |
| JPX Tokyo Stock Exchange | 2022 | A (横截面) | — (未核实*) | common factor / stock-relative | — | — | MSC 无 cross-sectional key, blocked |

\* 子代理调研失败未核实项, 不影响 Top 5 决策。

---

# 3. SOTA Method Matrix

| 方法 | 最早/核心论文 | 年份 | 方法类型 | 原问题 | 公开效果 | 与 RealMLP 区别 | MSCapital 实现方式 | 优先级 |
|---|---|:---:|---|---|---|---|---|---|
| **Supervised AutoEncoder** | JS 2021 1st (Kaggle writeup, L3) | 2021 | 新隐变量 | 匿名特征预测 | **单模型 6022.202 = 1st** (比赛) | 编码器学习 latent, 监督注入 | 152特征 → 32-64d latent concat 原特征 → RealMLP; 联合训练防泄漏 | **P0** |
| **AE 合成特征** | DRW 2025 1st (Kaggle writeup) | 2025 | 新隐变量 | 匿名特征预测 | 1st private 0.13959 (比赛) | 同上 | 同上 (两者是同一思想, 可合并实现) | **P0** |
| **市场状态条件化** | 自研 E02 (gate=True) + AdaRNN 思想 | 2026 | 新泛化机制 | 漂移下残差可预测性 | 残差 cos +0.0132, CI 不跨 0 (自研) | 建模对象从"行"变"状态×行" | 状态向量 (波动/深度/事件率 z) → 条件化残差模型 | **P0** |
| LOBERT | arXiv:2511.12563 | 2025 | 自监督事件表示 | LOB 消息级预训练 | 论文自报 (无比赛) | 事件级 token 序列 vs 窗口 summary | 逐事件流 → TinyLOBERT → 32/64d latent → RealMLP | P1 |
| OrderFusion | arXiv:2502.06830 | 2025 | 空间-时间表示 | 电力市场订单簿预测 | 论文自报 (电力, 非股票) | 2.5D 网格 vs 1D summary | 价格离散化 → 网格 → 轻量 CNN encoder | P1 |
| Neural Hawkes | arXiv:1612.09328; LOB: 2502.17417 | 2016/2025 | 事件生成机制 | 事件强度预测/模拟 | 论文自报 (做市模拟) | 显式 λ(t) 机制 vs 静态特征 | 事件流 → NHP → λ 特征 + 下事件时间 → RealMLP | P2 |
| HLOB | arXiv:2405.18938 | 2024 | 结构表示 | LOB 分类/回归 | 论文自报 | 图结构 vs summary | 与 M02 重叠风险高; 只偷 shape 统计 (已做 M02) | P2 |
| TabM | ICLR 2025, arXiv:2410.24210 | 2025 | tabular learner | 表格学习 | JS-RT 58th 实战 lb 0.0077 (比赛, top 5.8%) | 参数高效集成 vs 单 MLP | 直接替换/融合 RealMLP | P2 |
| Online Learning | JS-RT 58th (Kaggle writeup) | 2025 | 新学习机制 | 实时回流 | lb +0.0006 (比赛) | 测试期更新 vs 静态 | **N/A: MSC 无实时标签回流** | 不做 |
| AdaRNN | CIKM 2021, arXiv:2108.04443 | 2021 | 领域适应 | 时序协变量漂移 | 论文自报 (无金融漂移实战) | 分布匹配 vs 静态 | 只借"状态匹配/重加权"思想 (并入 P3-02) | P2 |
| TabPFN 2.x | arXiv:2501.02945 | 2025 | foundation | 时序预测 | 论文自报 (Chronos 对比) | 预训练模型 | 无金融实战证据, 且 1.2M 行太大 | 不做 |

---

# 4. Top 5 Method Cards

## 4.1 P3-01 — Supervised AutoEncoder latent 特征 (SAE)

### 来源
Jane Street 2021 1st (Yirun Zhang/VECTOR, writeup L3 公开) + DRW 2025 1st (A_A, AE 8 特征) + 学术血缘: SimLOB (2406.19396), LOBench (2505.02139)

### 原始问题
匿名表格特征 → 预测金融目标。冠军发现: 人工特征之外的 latent 仍有信息。

### 核心机制
```
输入 152+ 特征 → [编码器 → 32-64d latent] → 与原特征 concat → MLP → 预测
                       └── 解码器重建输入 (MSE) + 监督 target 注入编码器
```
- **监督注入**: target 参与编码器训练 (强制 latent 携带目标相关信息 + 梯度捷径)
- **防泄漏关键**: AE 与 MLP 在每个 CV split 内联合训练 (Yirun 原话: 单独预训练 AE 会造成 label leakage, 这是他从前人方案学到的核心修正)
- 配套: Gaussian noise 增强, swish, BN+Dropout, 3 seed 平均

### 为什么可能适合 MSCapital
- 两个独立冠军 (不同比赛、不同年代) 殊途同归 → 信号真实
- 项目 canonical OOF 的 inner split 结构 = SAE 联合训练防泄漏的现成基础设施
- 与 E01 ReVol-lite 的 "residual-scale" 特征 (四折全正) 可能叠加

### 它提供了什么当前 152 dynamics 没有的信息
- 非线性组合特征: 152 个特征是手工线性/幂次组合, AE 能学任意高阶非线性交互
- 重建目标强制 latent 保留"完整市场信息"而非只保留 target 信息 → 与 RealMLP 相关性可能低于人工特征 (融合价值)

### 最小实现版本 (1-2 天)
1. 取现有特征表 (152 dynamics 或 Clean Baseline 特征集), 在 canonical OOF inner split 内:
   - 每折: 训练 SAE (编码器 2 层 → 32d latent, 重建 + target 双 loss), 用 latent concat 原特征训练 RealMLP (或轻量 MLP)
2. 输出: 四折 outer 分数 + ΔPSEUDO
3. 变体: latent 维度 {16, 32, 64}, 是否 concat 原特征 {是, 否}

### Full version
- 在 Kaggle P100 上用全量 125 万行训练, 256d latent + 更强的编码器 (swish/GN)
- latent 同时加入表格融合栈和 lb142 融合
- 与 E01 residual-scale 特征叠加

### 成功门禁
```text
ΔPSEUDO >= +0.0015
或: 接近 baseline 但 prediction corr < 0.80 (独立信息)
```

### 风险
- 过拟合: 依赖联合训练纪律, 单独预训练必泄漏 (冠军原话)
- 与 152 特征信息重叠 → 增量小 (但 corr 门禁兜底)
- 计算: 云端 GPU 需要, 本地 CPU 慢 (可先用 40 万行子集试)

---

## 4.2 P3-02 — 市场状态条件化 (Reconditioning v2)

### 来源
自研 E02 (gate=True, 2026-08-13) + 思想祖先: AdaRNN (2108.04443), Optiver RV 1st KNN, Hull 4th 组合工程

### 原始问题
月度漂移下, 残差的可预测性不是常数——某些市场状态下残差更可预测。

### 核心机制
```
状态向量 (ReVol-lite 11 特征: 波动/深度/事件率/流量 z-score)
   → 残差目标 r = y − β·ŷ_baseline
   → 条件化模型: E[r | 状态, 特征] ≠ 0
```
E02 已验证: 用历史状态预测残差, 残差 cosine +0.0132 (bootstrap CI [+0.0084, +0.0185] 不跨 0)。

### 为什么可能适合 MSCapital
- 唯一 gate=True 的自研方向
- 直接攻击核心痛点: 漂移 (adversarial AUC 0.77)
- E03 揭示的 "top-3 月份集中" 表明信号随月份波动 → 状态条件化正是解法

### 它提供了什么当前 152 dynamics 没有的信息
- 新建模对象: 从"行级特征"升级为"状态 × 行级特征"的交互
- 市场状态本身 (波动 regime) 是 152 特征未显式建模的维度 (E02 的 11 状态特征全是残差尺度的)

### 最小实现版本 (1-2 天)
1. E02 已有: HistGB 池化残差预测 + bootstrap CI
2. v2: 状态特征 × 残差模型升级为 CatBoost 残差模型 (带状态特征作输入) + 与 baseline 融合
3. 关键验证: 四折 outer 的 ΔPSEUDO 是否 ≥ +0.0015; 检查状态交互的增益是否集中在特定月份 (E03 教训)

### Full version
- 状态分层模型: 按状态聚类训练多个 specialist (类似 recent-regime specialist, 内部假设已标注)
- 状态特征进入 RealMLP 主模型 (concat), 而非只做残差模型

### 成功门禁
```text
ΔPSEUDO >= +0.0015 且至少 3/4 outer 为正
```

### 风险
- E01 gate False 的教训: 四折全正仍可能不达门禁
- E03 集中度风险: 增益可能集中在少数月份 → 必须查 monthly breakdown
- 状态特征本身可能过拟合 (E02 用 ridge/HistGB 池化, v2 换成强模型时注意)

---

## 4.3 P3-03 — TinyLOBERT 掩码事件预训练

### 来源
LOBERT: arXiv:2511.12563 (2025-11, "Generative AI Foundation Model for LOB Messages") + 血缘: 事件序列 transformer (2201.00044), Deep OFI

### 原始问题
LOB 消息级建模: 不规则事件时间、快速 regime 切换、高频交易者反应。

### 核心机制
- 消息级 tokenization (每条 order/transaction 事件 = token, 连续值编码 + 时间编码)
- 掩码事件/消息预测 (masked event modeling) 自监督预训练
- encoder-only transformer → 序列级 latent

### 为什么可能适合 MSCapital
- 表示层级完全不同: 现在是"60 秒窗口 → 聚合 summary", LOBERT 是"60 秒窗口 → 事件序列 → 序列 latent"
- 保留事件顺序/时序结构 (152 特征丢失了顺序信息的大部分)
- 自监督 → 不依赖标签, 预训练后可冻结 latent 给 RealMLP

### 它提供了什么当前 152 dynamics 没有的信息
- 事件序列的**顺序与条件依赖** (哪个事件跟在哪个事件后面)
- 不规则时间间隔的显式建模 (seconds_before_predict 分布)

### 最小实现版本 (3-5 天)
1. 用逐事件 order.feather (云端) 构建 token 序列: 每样本截取窗口内事件, 排序, 编码 (price/volume/side/action + 相对时间)
2. 小模型: 2-4 层 transformer, d_model 64-128, 掩码率 15%
3. 预训练后取 [CLS]/mean-pool → 32/64d latent; 与 152 特征 concat → RealMLP
4. 验证: 四折 outer ΔPSEUDO; latent 与 152 特征的 corr 诊断

### Full version
- 更大模型 + 更长预训练 + 时间编码消融
- latent 参与 lb142 融合

### 成功门禁
```text
ΔPSEUDO >= +0.0015
或: 接近 baseline 且 corr(latent, 152features) < 0.70
```

### 风险
- 论文证据弱 (无比赛验证); 事件级 transformer 训练成本高
- P100 (sm60) 兼容性 (torch 降级经验已有)
- 序列长度控制: 每样本 ~218 事件, 需要截断/采样策略
- **最大的隐性风险**: 秒级聚合数据已丢失事件顺序 → 需用云端原始数据, 本地无法快速迭代

---

## 4.4 P3-04 — OrderFusion 式 2.5D 时空网格编码

### 来源
OrderFusion: arXiv:2502.06830 (2025, 电力市场订单簿) + 血缘: DeepLOB (1808.03668), JPM robust repr (2110.05479)

### 核心机制
- 把订单簿编码为 2.5D 网格: 价格档 (空间) × 时间 × 深度/方向
- bid/ask 交互 + cross-attention 建模档位间关系
- 端到端概率预测 (原论文为电力日内价格)

### 为什么可能适合 MSCapital
- 显式建模"价格档 × 时间"结构 (152 特征是扁平 summary)
- 可迁移成轻量 grid encoder: 只取网格 + 小 CNN/attention → 特征给 RealMLP (不端到端)

### 它提供了什么当前 152 dynamics 没有的信息
- 档位间的空间结构 (L1/L2 深度随时间的联合演化)
- 价格离散化后的位置信息 (当前特征用连续价格, 档位关系被弱化)

### 最小实现版本 (3-5 天)
1. 事件流 → 重建价格网格: 价格按 tick 离散化 (观察价格网格分布后定 bin), 时间 60×1s
2. 轻量 encoder: 2 层 CNN 或 1 层 cross-attention → 16/32d grid latent
3. latent → 与 152 特征 concat → RealMLP; 或单独残差模型 (复用 M01-A 残差协议)

### Full version
- 完整 2.5D grid + 端到端训练 (云端 GPU)
- grid latent 与 SAE latent 融合

### 成功门禁
```text
ΔPSEUDO >= +0.0015
```

### 风险
- **领域差异**: 论文是电力市场 (订单簿稀疏、tick 结构不同), 迁移需重新设计网格
- 与 M02 (Market-Centered Geometry) 信息重叠风险 (M02 四折正但 +0.0003)
- 价格离散化 bin 选择敏感; 数据只有 L1/L2 深度信息 (事件流重建)

---

## 4.5 P3-05 — Neural Hawkes 事件强度特征

### 来源
Neural Hawkes: arXiv:1612.09328 (Mei & Eisner 2016); LOB 应用: arXiv:2502.17417 (2025, 做市模拟)

### 核心机制
- 多元点过程: 显式建模 λ_buy_add, λ_sell_add, λ_buy_cancel, λ_sell_cancel, λ_buy_trade, λ_sell_trade
- 神经自调制: 事件历史 → 隐状态 → 各事件类型强度 + 下一事件时间
- 输出: 预测的时刻各事件强度 + 期望下一事件时间

### 为什么可能适合 MSCapital
- 提供"未来事件机制"信息: 不是过去 60 秒的统计, 而是对未来事件发生概率的预测
- 2502.17417 验证了 12 类 LOB 事件的 NHP 建模可行性

### 它提供了什么当前 152 dynamics 没有的信息
- 条件强度 (hazard) 的未来演化 (152 特征全是历史窗口统计)
- 事件间条件依赖 (某类事件后另一类事件概率上升)

### 最小实现版本 (5-7 天)
1. 事件流 → (类型, 时间) 序列; 训练小型 NHP (或简化: 6 类事件的多任务强度 MLP, 用 log-likelihood)
2. 在预测时刻 t 输出: 6 个强度 + 下一事件时间 + 隐状态 16-32d
3. 强度特征 → RealMLP / 残差模型
4. 关键诊断: 与 M01-A 事件率特征的相关性 (若 >0.95 说明无增量, 直接终止)

### Full version
- 完整 Neural Hawkes (自调制 LSTM) + 强度轨迹特征 (预测窗口内强度曲线)
- 强度误差作为异常检测特征

### 成功门禁
```text
ΔPSEUDO >= +0.0015, 且强度特征与 M01-A 事件率 corr < 0.90
```

### 风险
- 训练成本最高 (事件级序列 MLE, 125 万样本 × ~218 事件)
- 信息可能与 M01-A (24 事件流特征) 高度重叠 → 先做相关性诊断再投入
- 论文证据是模拟/做市应用, 非预测比赛

---

# 5. Method Genealogy

## 微观结构线 (表示演化)

```text
Cont OFI (2014) — 买卖订单流 ≠ 静态量
    ├── MLOFI (2019) — 多档订单流
    ├── Deep OFI (2023) — stationary flow + 简单 ANN > raw LOB
    └── 事件动力学表示 (本项目 152 特征, 2026) — LB +0.010 ← 项目现在的位置
            │
            ▼
    Representation Learning 分岔
    ├── HLOB (2024) — 盘口结构/图
    ├── LOBERT (2025) — 消息级掩码预训练 → P3-03
    ├── OrderFusion (2025) — 2.5D 时空网格 → P3-04
    ├── Neural Hawkes (2016→2025) — 事件强度机制 → P3-05
    └── Supervised AE / SimLOB / LOBench — latent 压缩 → P3-01
            │
            ▼
       简单强 learner (RealMLP / TabM) — 已验证: 表示 > 架构
```

## Kaggle 实战线 (证据演化)

```text
JS 2021 1st: Supervised AE + MLP (单模型夺冠) ──┐
DRW 2025 1st: AE 8 特征 + MLP (SGD/混合loss) ───┼─→ P3-01 (双冠军印证)
JS-RT 2024 58th: TabM + AE-MLP + online ────────┘    (online → MSC N/A)
                                                    
Optiver RV 1st: KNN 历史状态 ──→ 项目 M05 (gate False)
Ubiquant 2nd: Robust CV ──→ 项目 canonical OOF 制度 (已吸收)
Optiver Close 1st: 后处理/合成指数 ──→ M05 交互特征 (gate False)
Hull 4th: 组合工程 > 预测精度 ──→ 思想: 特征不预测也能降噪
                                                   
自研线: M01-M05 (+0.0002~0.0005) → E01 (+0.0011, 最接近) → E02 (gate=True) → P3-02
```

**核心演化结论**: 两个源头 (微观结构表示学习 + Kaggle 冠军实战) 在过去 5 年汇聚到同一结论——**latent/结构表示 + 简单强 learner**。项目已验证了表示的第一代 (152 特征 → +0.010), 第二代是"让模型自己学表示" (SAE/LOBERT/网格)。

---

# 6. 最终实验队列

```
P3-01
方法: Supervised AutoEncoder latent (32-64d) concat 原特征 → RealMLP; canonical OOF inner split 内联合训练防泄漏
来源: JS 2021 1st (L3 notebook) + DRW 2025 1st (AE 8 features)
代码工作量: 中 (1-2 天; 复用 canonical OOF 基础设施 + 现有特征表)
GPU: Kaggle P100 (正式版); 本地 CPU 40 万行子集可先试
最小实验: 每折训练 SAE (编码器2层→32d, 重建+target 双loss) → latent concat 152特征 → RealMLP → 四折 outer
成功门禁: ΔPSEUDO >= +0.0015; 或 corr(latent, 152) < 0.80 且分数接近 baseline
下一步: 256d + 更强编码器; latent 进 lb142 融合; 与 E01 residual-scale 叠加

P3-02
方法: 市场状态条件化 v2 — 状态特征 (E02 的 11 个 ReVol-lite) × 残差模型升级为 CatBoost, 状态进主模型 concat
来源: 自研 E02 (gate=True) + AdaRNN 思想 (2108.04443) + Hull 4th 组合思想
代码工作量: 低 (1 天; E02 代码已有)
GPU: 不需要 (本地 CPU)
最小实验: E02 → CatBoost 残差模型 + 状态特征; monthly breakdown 检查 (E03 教训); 四折 outer
成功门禁: ΔPSEUDO >= +0.0015 且 ≥3/4 outer 正, 且增益不集中 top-3 月份
下一步: 状态聚类 specialist 模型; 状态特征进 RealMLP 主模型

P3-03
方法: TinyLOBERT — 事件序列 transformer, masked event modeling 预训练, 取 latent → RealMLP
来源: LOBERT arXiv:2511.12563 (2025-11)
代码工作量: 高 (3-5 天; 需构建事件 tokenizer + 预训练 + 下游)
GPU: Kaggle P100 (必须; 本地无原始逐事件数据)
最小实验: 2-4 层 transformer (d=64-128), 掩码率 15%, 预训练 1 epoch → 32d latent → concat 152 → RealMLP (PSEUDO)
成功门禁: ΔPSEUDO >= +0.0015; 或 corr(latent, 152) < 0.70 且分数接近 baseline
下一步: 更大模型/更长预训练; 时间编码消融; latent 进融合栈

P3-04
方法: OrderFusion 式 2.5D 网格 — 价格离散化 × 60s 时间 × 深度, 轻量 CNN/cross-attention encoder → grid latent
来源: OrderFusion arXiv:2502.06830 (电力市场, 需重新设计网格)
代码工作量: 中 (3-5 天)
GPU: 可选 (小网格 CPU 可跑)
最小实验: 网格重建 + 2 层 CNN → 16-32d → concat → RealMLP; 与 M02 特征做 corr 诊断
成功门禁: ΔPSEUDO >= +0.0015, 且与 M02 corr < 0.85
下一步: 完整 2.5D + 端到端; grid latent 与 SAE latent 融合

P3-05
方法: Neural Hawkes 事件强度 — 6 类事件 λ + 下一事件时间 + 隐状态 → RealMLP
来源: Neural Hawkes arXiv:1612.09328; LOB: arXiv:2502.17417
代码工作量: 高 (5-7 天; 事件级 MLE)
GPU: Kaggle P100
最小实验: 先做相关性诊断 (NHP 强度 vs M01-A 事件率 corr, >0.90 直接终止); 简化版 (6 类多任务强度 MLP) 验证可行性
成功门禁: ΔPSEUDO >= +0.0015 且强度与 M01-A corr < 0.90
下一步: 完整自调制 NHP; 强度轨迹特征
```

---

# 7. 判断汇总

## 今天只选两个开始写代码

**P3-01 (SAE) + P3-02 (状态条件化)**。
- P3-01 是外部证据最强的一条路 (两个独立冠军), P3-02 是内部证据最强的一条路 (唯一 gate=True)。
- 两者正交: 一个换"表示", 一个换"建模机制"——即使都只加 +0.001, 叠加也可能到门禁。
- 都是 1-2 天工作量, 都复用现有基础设施。

## 看起来先进但当前不值得做

| 方法 | 淘汰原因 |
|---|---|
| TabM | learner 改进 (非表示); JS-RT 证据只是 top 5.8%; 与 RealMLP 架构重叠风险高; 优先级低于 P3-01 |
| Online Learning | **MSC 无实时标签回流** (离线提交), 机制不适用; 伪在线 (recent-refit) 等价于窗口选择, 低优先 |
| TabPFN 2.x | 无金融实战证据; 1.2M 行 × 大模型, 成本高 |
| AdaRNN 全套 / GroupDRO / CORAL / domain adversarial | 无真实金融漂移数据实证 (论文多为合成/图像); 只借思想 (已并入 P3-02) |
| HLOB 全套网络 | 与 M02 重叠; 数据只有 L1/L2, 图结构空间有限; 只保留 shape 思想 (M02 已做) |
| DeepLOB / TransLOB / 大 Transformer | raw sequence 路线, TCN OOD 教训 (LB 0.082) |
| 更多特征小变体 / OFI 变体 / 调参 | 用户已明确排除; M01-M05 已证边际 < +0.0005 |

---

# 8. 证据可信度说明

- **已逐条核实**: JS 2021 1st writeup (本机抓取, 353 votes, L3 notebook), DRW 2025 1st/2nd writeups (本机抓取), JS-RT 2024 58th writeup (本机抓取), Hull 4th writeup (本机抓取), 全部 arXiv 编号经 API 验证 (LOBERT 2511.12563, OrderFusion 2502.06830, NHP 1612.09328/2502.17417, TabM 2410.24210, AdaRNN 2108.04443)。
- **未核实 (子代理失败)**: Ubiquant 1st/3rd 细节, G-Research top 细节, JPX top 细节, DRW 3rd-10th 方案细节。这些不影响 Top 5 决策 (Top 5 全部基于已核实证据)。
- **数字来源标注**: 比赛数字 (private LB) 来自 Kaggle 榜单/writeup; 论文数字均标注"论文自报"。
