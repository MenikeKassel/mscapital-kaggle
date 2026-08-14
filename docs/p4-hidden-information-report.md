# P4 Hidden Information Investigation Report

> 日期: 2026-08-14 | 阶段: P4 隐藏信息层调查
> 核心问题: 0.159/0.155 vs 0.135/0.142 的差距来自哪里?
> 方法: 数据取证 (本机原始数据) + LB142 源码逆向 + Kaggle 社区情报 + 协议实验

---

# 1. Executive Conclusion

**0.135 → 0.155~0.159 的差距,最可能来自:**

## H4(概率 45%):外部模型使用了我们从未建模的 600 秒盘口快照流 + 未知 factors 表

**证据**:
- 本机原始数据 `market.feather` 每样本 ~189 行 L1/L2 快照,覆盖 **600 秒**(199.5M 行在 60-600s 历史区),我们从未建模(152 特征只用 60s order/tx;M02 只取每样本最后一行)
- **LB142 源码逆向**:v9 网格模型 market 流 = 200/400 步 × 12 通道——正是这个快照流;v10 用 `best_factors` 表(内容未知)
- **Forensics 实锤**:ref−v7 分歧最大的 5% 样本 = 高活动(订单数 2.6×)、高波动(2.3×)、强自相关(3×)、低缺失(0.38 vs 0.67)——**LB142 在信息量大的样本上掌握了额外信息**
- **P4-04 深化**:分歧不是噪声——**随活动度单调的系统性方向偏差**(高活动样本 ref 相对 v7 偏正,低活动偏负);corr(|d|, 活动/波动特征) = 0.26~0.36;且高 |d| 样本 = 高 |target| 样本(因为 |target| 随波动率缩放)

**反证**: 我用 600s 快照的**聚合特征**(34 个)跑协议 → PSEUDO -0.0002 无信号;可能聚合丢序列信息,或信息来自 factors/训练技巧
**潜在收益**: 0.005~0.015(如果序列信息真实)
**验证成本**: 中(先做信息存在性诊断,再考虑序列建模)

## H3(概率 30%):target 是未归一化的未来收益,含月度市场漂移分量

**证据** (P4-03 取证):
- **|target| 随波动率单调递增 2.6 倍**(decile 0.00115→0.00300)→ target **未按波动率归一化**,是收益类量
- **月度均值显著漂移**:max |monthly mean|/se = **16.5**(month 70 均值 +0.00035)→ target 含时间/市场分量,非去均值量
- 月度 std ratio 2.69(0.00186~0.00500)
- 价格被归一化到 1.0 附近(离散 tick)
- R²(y ~ 窗口代理) = 0.00016 → target 未来主导,窗口信息不可直接解释

**反证**: corr(y, 窗口代理) ≈ 0(无直接窗口结构)
**验证成本**: 低(月均值后处理、分层评估)

## H5 null(概率 15%):主要是更强模型 + 训练技巧

**证据**: LB142 = 三流网格 CNN+attention + RealMLP×8 成员 + cosine loss + online BN 适配 + demean;bestwater 的 Transformer CV 0.1549 但 LB 0.120 说明架构本身不是答案
**验证成本**: 低(可部分通过复刻 LB142 训练管线测试)

## H1+H2(概率 10%):pseudo-asset / 跨样本序列

**证据**: 单一主资产(价格 1.0 簇);sample_id 顺序无边界连续性(ratio 0.9986);存在 0.5 价格第二簇(0.5% 样本,全月份均匀,**target 幅度仅主群 17%**——不是低波动资产能解释,可能是另一标的)
**反证**: 价格归一化抹平了跨样本连续性信号,资产身份即使存在也难以恢复

---

# 2. Missing Information Map

```text
                     MSCapital target
                          │
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
  local window        hidden identity     cross-row
  information         / regime            structure
        │                 │                 │
  60s order/tx       asset/time 600s    sequence/history
  (152 features)     snapshot stream     (cross-sample)
        │                 │                 │
     RealMLP         LB142 v9 grid     currently ?
    (0.134)          (200/400步×12ch)      ❌ 无证据
                          │
                    factors v10 (未知)     ← 真正的空白象限
```

**空白象限**:
1. ❌ **600s 盘口快照序列** —— LB142 已使用,我们未用 (P4-01)
2. ❌ **factors 表内容** —— LB142 v10 的输入,完全未知 (P4-02)
3. ❌ **target 归一化参考** —— 价格归一化暗示存在,未恢复 (P4-03)
4. ❌ **第二价格水平群 (0.5)** —— 可能是第二资产,未建模 (P4-04)

---

# 3. Hidden Structure Investigation

## asset identity (H1)
- **Evidence For**: 0.5 价格水平第二簇 (~6600 样本,全月份均匀分布, target 波动率 0.0017 < 全局 0.0026)
- **Evidence Against**: 99.5% 样本价格在 1.0 附近单峰;90% 相邻差为 0(离散 tick);per-sample 量级一致
- **Confidence**: 低-中 (第二簇存在,但占比 0.5%,影响有限)
- **Next experiment**: P4-04 第二簇深度分析 (order/tx 特征差异、target 结构、时间分布)

## time/session identity (H2a)
- **Evidence For**: 数据有 month 字段 (0-70);adversarial AUC 0.77 证明月漂移;market 快照有 600s 时间结构
- **Evidence Against**: 无 day/hour 字段;sample_id 顺序无连续性
- **Confidence**: 中 (月份已知;日/会话不可恢复)
- **Next**: 无 (月份已用于所有切分)

## sample adjacency (H2b)
- **Evidence For**: 无 — consecutive sample_id 边界跳变 = 随机配对 (ratio 0.9986)
- **Evidence Against**: 价格 per-sample 归一化到 1.0,抹平连续性;无事件 ID 连续性
- **Confidence**: 低 (跨样本序列重建在当前数据形态下不可行)
- **Next**: 无 (除非发现逆归一化方法)

## train/test mapping
- **Evidence For**: 无 (test 无 market 数据验证;processed test 结构同 train)
- **Evidence Against**: 无泄露讨论证据;前排提交次数少
- **Confidence**: 低
- **Next**: 无

---

# 4. Target Reverse Engineering (Top 5 hypotheses)

| # | 假设 | 金融含义 | 测试方法 | 所需字段 | 预期特征 |
|---|---|---|---|---|---|
| T1 | 未来 mid 收益 (归一化) | y ≈ (mid_{t+h}−mid_t)/mid_t | corr(y, 窗口内收益代理);与波动率关系 | market (mid) | y 与窗口内趋势相关,std ∝ 波动率 |
| T2 | 未来 VWAP 收益 | 加权平均价变动 | 构造窗口 VWAP,检查 y 的尺度 | market (transaction_avgprice) | y std 与 VWAP 波动一致 |
| T3 | 市场中性/去均值收益 | y = r − mean(r) 某参考 | 检查 y 的均值结构 (demean 影响) | label | y 均值≈0 (已观察: −1.2e-5) |
| T4 | 未来 OFI/流量冲击 | 订单流不平衡的未来影响 | corr(y, 窗口 OFI) | order | 高活动样本 |y| 更大 (forensics 已见!) |
| T5 | 未来波动/极值 | 价格极值行程 | corr(y, |y|, 波动率代理) | market | |y| 与波动率正相关 |

**注意**: forensics 已发现高活动样本上 LB142 分歧最大——与 T4 一致 (订单流冲击在活动期最强)。

---

# 5. LB142 Forensics

**方案** (已执行第一轮):
```text
d = unit(pred_lb142) − unit(pred_v7)   (test, 647,896 行)
```

**第一轮结果** (p4_lb142_forensics.py):
- corr(ref, v7) = 0.82 (与训练期一致)
- **高 |d| (top 5%) 样本画像**:
  | 特征 | 高 |d| | 低 |d| | 比率 |
  |---|---|---|---|
  | 45s 订单数 o_n_45 | 271 | 106 | **2.6×** |
  | 价格波动范围比 | 0.0034 | 0.0015 | **2.3×** |
  | 秒级自相关 lag1 | 0.396 | 0.126 | **3.1×** |
  | 最大时间间隔 | 5.96 | 11.59 | 0.51× |
  | 缺失率 x_sec_t_max | 0.38 | 0.67 | 0.56× |

**结论**: LB142 与我们的分歧集中在**高活动、高波动、事件密集、数据完整**的样本。两个解释:
(a) LB142 的 600s market 流在活动期捕捉到我们没有的信息 (最优解释);
(b) 我们的 60s order/tx 特征在高活动期饱和,而 LB142 的网格不过饱和。

**下一步**: 对高 |d| 样本做 market 600s 特征对比 (确认 (a)),并检查 factors 表是否可获取。

---

# 6. Leaderboard Intelligence

| 类别 | 内容 | 来源 |
|---|---|---|
| **已证实** | 前排 0.150+ 队伍**提交次数很少** | iamltpn (7th) 讨论回复 |
| **已证实** | bestwater (35th): LGB Stacking 50% + PB Blend 50% = LB 0.139;90 特征 CV 0.1389/LB 0.119 | bestwater 讨论帖 |
| **已证实** | yunsuxiaozi (49th): 80W/40W 切分,CV-LB gap ≈ 0.01,代码已开源 | 讨论回复 |
| **已证实** | MAIDANG (8th): 单指标 CV 过拟合;多泛化指标 | 讨论回复 |
| **强间接** | 主办方 Rib~ 推荐参考 Optiver RV / TATC (特征工程) + 传感器/波形类比赛 (序列建模) | host 讨论帖 |
| **社区猜测** | CV 与 LB 在 0.150 以下一致,以上可能不一致 | iamltpn |
| **无证据** | 无泄漏/隐藏 ID/行序重排的公开讨论 | — |

**判断**: 无作弊/泄漏证据。0.155+ 梯队 = 真实方法差异,与 H4 (未建模信息层) 一致。

---

# 7. Final Experiment Queue

```text
P4-05  月度漂移分量建模 (target 月均值后处理)  [NEW, 最高性价比]
假设: target 含月度市场漂移 (max |mean|/se=16.5), 我们未显式建模
实验: (a) 用历史月份均值 (滚动) 预测未来月均值, 加到 baseline 预测;
      (b) 月度固定效应: baseline 残差按月的均值回归;
      (c) 活动度分层评估: 高活动样本 (top 10% 活动度) 的 PSEUDO 单独计算
需要的数据: canonical OOF + baseline (本机)
代码工作量: 0.5 天 (纯后处理, 无模型)
成功判据: (a)(b) ΔPSEUDO >= +0.0005 且跨 fold 稳定; (c) 揭示分层弱项
失败后: target 月均值不可预测 → 漂移是噪声, H3 的市场分量部分关闭
下一步: 若 (c) 显示高活动弱项 → 高活动样本的 600s 历史信息 (P4-01)

P4-01  market 600s 快照流信息存在性 (针对高活动样本)
假设: LB142 的独立信息来自 600s 盘口快照序列, 且集中高活动样本
实验: (a) 高|d| vs 低|d| 样本的 market 600s 特征对比 (波动率/深度演化/趋势);
      (b) target 与 market 历史量 (600s 波动/趋势) 的直接相关 (按活动度分层);
      (c) 若 (a)(b) 为正: market 快照序列的轻量时域签名特征 → M01-A 协议
需要的数据: market.feather (本机), lb142 ref, v7 pred (本机)
代码工作量: 1-2 天
成功判据: (a) 高|d|样本 market 特征显著差异; (b) 分层 |corr| > 0.01 跨 fold 稳定;
          (c) ΔPSEUDO >= +0.0015
失败后: H4 的序列部分排除, 转向 factors 表 (P4-02)
下一步: 序列建模 (LB142 式 grid, 需先获得 grids 或自建)

P4-02  LB142 v10 factors 表逆向
假设: v10 的 factors 包含我们 152 特征没有的量
实验: 从包内权重/缩放器 (scaler_median/factors) 反推特征尺度分布;
      检查 f0726 特征与 v10 权重的对齐;搜索 factors 是否在 Kaggle 数据集公开
需要的数据: lb0142 包 weights/ (本机)
代码工作量: 1 天
成功判据: 识别出 v10 特征集与 152 特征的差异 (重叠率 < 80%)
失败后: factors 不可得 → 只能通过复刻 LB142 推理验证

P4-03  target proxy 相关性 (H3 深化)  [部分完成]
假设: target 是未归一化未来收益
实验: 已完成: |y|~波动率 2.6x、月均值漂移 16.5se、R2=0.00016、0.5群 |y| 17%
      待补: (a) target 与 order/tx 未来事件代理的关系; (b) target 尾部与活动度
需要的数据: 本机
代码工作量: 0.5 天
成功判据: 已达成 (指纹确认)
失败后: N/A

P4-04  0.5 价格群深度分析 (第二资产?)
假设: 0.5 簇是第二资产/独立市场状态 (|y| 仅主群 17%)
实验: order/tx/market 特征对比 (1.0 簇 vs 0.5 簇);target 结构;月份-时间分布;
      能否用特征聚类稳定复现该群 (门禁: 聚类 > 90% 一致)
需要的数据: 本机全部
代码工作量: 0.5 天
成功判据: 0.5 簇在 order/tx/market 特征上显著分离且跨月稳定
失败后: 群是噪声/归一化伪影 → H1 关闭
下一步: 若成立 → per-asset 特征/模型
```

---

# 8. 如果今天只投入两个实验

**P4-01 (market 600s 序列信息存在性)+ P4-03 (target proxy 相关性)**。

- **P4-01**: H4 是目前证据最强的假设 (forensics 分歧画像 + LB142 源码),而 market 600s 快照流是唯一被证实存在、被外部模型使用、而我们从未建模的信息层。**直接对答案**——如果 target 与 600s 历史量有结构关系,或高 |d| 样本在 market 特征上显著分离,就找到了 0.135→0.155 的机制。
- **P4-03**: 0.5 天工作量,回答"target 到底是什么"——如果是相对量,整个建模框架要调整 (demean/相对特征);如果是绝对收益,则 H3 关闭,资源集中到 P4-01。

两者都比继续优化 E01 (+0.00110 且月份集中) 更可能解释 0.02 量级的差距,因为 E01 只优化残差尾部,而 P4-01/P4-03 直接检验"我们是否在预测错误的问题"。

---

# 9. 证据可信度

- ✅ 本机实测: market.feather 结构/秒范围/行数、价格分布、H2 邻接性、forensics 画像
- ✅ LB142 源码逐文件阅读 (dataset/gridcfg/online/metrics/members/manifest)
- ✅ Kaggle 讨论实页抓取 (bestwater/iamltpn/MAIDANG/yunsuxiaozi/Rib~)
- ⚠️ factors 表内容不可得 (精简包排除);LB142 网格构建脚本不可得
- ⚠️ 0.155+/0.159 队伍方法无公开信息 (仅提交次数情报)

---

# 10. P4-05 / P4-01a / P4-04b 执行结果 (2026-08-14)

## P4-05 月度漂移可预测性 → **失败 (决定性负结果)**

- μ_m 序列 (0-70): ACF(1..12) 全部 |r|<0.25 (近白噪声);sign persistence 0.471 (< 随机 0.5);趋势 7e-6/月
- 8 种估计 (expanding/roll3/6/12/ewma3/6/12/trend) × 4 outer, 严格 rolling-origin
- **全部 ΔPSEUDO ≈ 0 (最大 +0.000001)**;α 选择器几乎全选 0;months_pos ≈ 随机水平
- **结论: "月均值显著" (16.5 SE) 与 "月均值可预测" 彻底分离 — 漂移存在但不可从历史预测, 关闭月均值后处理路线**

## P4-01a 600s 长上下文测试 → **通过 (H4 升级为直接证据)**

d_orth = ref − E[ref|v7] (test, n=647,896, std=0.000638):

| 预测器 | R² (d_orth) |
|---|---:|
| M_short (仅最后 60s) | 0.00093 |
| **M_long (仅 -600..-60s)** | **0.00117** (permutation null max=0.000063 → **18.4×**) |
| M_short + M_long | 0.00139 (long increment +0.00046) |
| 最远段 m600_300 (-600..-300s) 单独 | 0.00065 (独立解释力!) |

- **超出最后 60s 的 540s market 信息显著解释 LB142-v7 分歧** → LB142 确实利用长时域 context
- 4 段 R² 均匀 (0.00065~0.00093) → 长历史信息分散在各时段, 不是 60s 窗口特征的另一种表达
- **下一步: 600s temporal-state compression 成为主线方向**

## P4-04b 0.5 价格群 → **H1' 重开 (第二 instrument/regime 证据)**

- train: 6252 样本, **71 个月全部存在** (每月 25-425)
- **test: 4900 样本 (0.76%) 同样存在** → 跨 train/test 稳定
- target: 均值 -0.00019, std 0.0017, **|y| 仅主群 17%** (scale discontinuity 无法用低波动解释)
- |d_orth| 略低于主群 (0.00033 vs 0.00042)
- **判定: 满足"跨所有月份稳定存在 + 独立指纹" → H1' (latent instrument/normalization bucket/source regime) 重开**

## 分叉逻辑更新

```text
P4-05 失败      → 不加 market-time prior (关闭)
P4-01a 成功     → 研究 600s long-context information extraction (主线!)
0.5 群稳定跨月  → 重开 H1' latent instrument/group 调查
```
