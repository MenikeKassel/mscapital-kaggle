# MSCapital 方法地图 (Method Map)

> 按"思想"分类的已尝试方法总览。✅ 有效 / 🟡 弱信号 / ❌ 已证伪 / ⚠️ 协议有问题 / 🧪 尚未测试
> 更新: 2026-08-15 (仓库工程化整理 Phase K)

## Tabular aggregation 表格聚合
```
├── LightGBM (官方参数)          ✅ B1, 全项目锚点
├── CatBoost                      ✅ 最稳健单模型 (P0-01), 但绝对弱
├── XGBoost                       🟡 与 LGB 相关 0.96, 互补有限
├── MLP 表格 (best-state)         🟡 温和超越 GBDT (+0.0019), 弱于 CatBoost
├── 三模型融合 (LGB+XGB+MLP)      ✅ G1-G3, 0.122 平台期 (饱和)
├── 152 特征 (f0726 复刻)         ✅ v7 突破核心
├── 22 微观特征 (P1-1a)           ✅ v5 +0.0036 最大单步
├── Z 创新特征 (SCFI)             ✅ P5-04/P5-06/P5-07, 双 learner 确认
└── 90 特征甜点位                 ✅ C1-FE 证伪更多特征
```

## Metric-aware 指标感知
```
├── cosine loss                    ⚠️ P4-10 INVALID (验证偏差); 生产禁混辅助 loss
├── MSE (标准)                     ✅ 生产默认
├── hybrid MSE+cosine              ❌ 4.14 实锤名义混合实际单任务
├── RMS 尺度校准融合               ✅ C-04 冻结规则
└── 全局尺度无关性                 ✅ F001 (浪费全局校准)
```

## Market sequence 市场序列
```
├── market-only 序列               🟡 P5-01 ~+0.0005 (大部分已在聚合特征)
├── 600s 长上下文                  ✅ P4-01a H4 直接证据
├── market history 聚合特征        🟡 P4-17 ~+0.0005
├── micro_price / spread_std 族    ❌ GAP-N1N3 (EDA 线性信号 → 残差零增量, PSEUDO +0.0000)
├── mid_range/mid_std 变体         ❌ P7-01 (对 cosine 无用)
└── TCN 序列模型                   ❌ v6 0.082 灾难 (N005)
```

## Conditional information 条件信息
```
├── market → 状态表示              🟡 E-01 ReVol-lite +0.0011 (gate 未过)
├── 状态条件化拼接                 ❌ P3-02 (稀释)
├── 订单流创新 (SCFI: O−E[O|M])   ✅ P5-04/P5-06/P5-07 — 当前最有效方向
├── 残差条件化 (Reconditionor)     ❌ E-02 (窗口内≠跨月)
├── FiLM / 条件 adapter            🧪 未测试 (完整版, 等 SCFI 系列消化)
└── 事件时条件强度 (ETCI)          🧪 未测试 (plan v1.9.0 队列)
```

## Residual modeling 残差建模
```
├── residual target (y−ŷ)          ❌ P5-03 (AUC 0.51 预判死)
├── OOF 残差均值                   ❌ P4-08, M-01 (+0.0000)
├── 检索-残差 (P6R FAISS KNN)      ❌ P6R-00 (gate 39%, KILL)
├── 局部回归 (P6R-01)              🧪 挂起 (等拍板)
└── 五连杀总结                     ❌ F011: 可解释≠可预测
```

## Gating / Amplitude 门控与幅度
```
├── 4-bin 幅度 gate                ❌ P5-03 (−0.000146 嵌套)
├── γ 幂族调制                     ❌ P5-03 (同判)
├── volatility confidence          ❌ P7-01 (α 单调=0)
├── 时间衰减加权                   ❌ E1-TW
├── 特征相对化                     ❌ P1-04 (第二轮)
└── "Market→Confidence" 假设       ❌ P7-01 证伪 (GPT1 round2 核心主张)
```

## Unsupervised / 表示学习
```
├── SAE 稀疏编码                   ❌ P3-01
├── TinyLOBERT 掩码预训练          ❌ P3-03 (corr 0.86~0.98)
├── 2.5D 网格投影                  ❌ P3-04
├── NHP/Hawkes 强度                ❌ P3-05 (|r|≤0.01)
├── Path Signature                 ❌ M-04
└── SSL 预训练                     🧪 未测试 (plan 第二阶段, 低优先)
```

## 形状 / 谱 (反演不变)
```
├── 短窗跨通道几何 (RICS)          ❌ P5-05 (≤0.011)
├── lag-cov / 相位                 ❌ P5-05 (R4 -0.006 反转)
├── SPCE 四臂 probe                🧪 未测试 (plan v1.9.0 队列, 信号被 P5-05 质疑)
└── TRIS shapelet                  🧪 未测试 (同上)
```

## 外部融合
```
├── RealMLP 复刻 (152 特征)        ✅ v7 +0.010
├── lb142 开源推理包               ✅ v8b 0.142 (#30)
├── TCN 融合                       ❌ v6 0.082
└── 外部方案直接融合               ✅ F015 合法且有效
```

## 隐藏信息面
```
├── 资产身份 (H1)                  ❌ P4-15
├── 时间身份 (H2)                  ❌ P4-15
├── 600s 流 (H4)                   ✅ P4-01a
├── 未知 factors (H4)              🟡 P4-02 (OFI 弱正)
├── target 逆向 (H3)               🟡 P4-03 未唯一确定
├── 月度漂移预测 (H3)              ❌ P4-05
└── LB142 分歧样本                 🟡 P4-16 (高波动集中)
```

## 未测试方向 (plan v1.9.0 剩余)
```
├── O→T lag response               ❌ P8-01A 证伪 (2026-08-15: 1s bin 跨秒响应不存在, 0/71月, placebo 全覆盖; 同期 lag=0 是机械重合)
├── SPCE 四臂 / TRIS random        🧪 低优先 (P5-05 已弱化谱线)
├── hard regime experts            🧪 低优先 (幅度路线已死, 需重新论证)
├── soft MoE                       🧪 依赖 P6-04
├── ETCI 事件时臂                  ⬇️ 降级低优先 (P8-01A 连带: 无 O→T 响应则 latency 类大概率是活动度特征)
└── P6R-01 终裁                    🧪 挂起等拍板
```

---

## P9-Lite 归因 (2026-08-20): cancel / event-time / M55
```text
├── 撤单侧拆不对称 Cancel Pressure (F1侧)  🟡 P9-A-LITE +0.0041 frozen (20/20聚合/13/20月Δ)
│     → 真实有效但 regime 集中: hi_act +0.0103 / low_act −0.0064 → 非 clean GREEN, 进联合/校准
│     → 与 Z 绿灯同源 (Z_ob_cancel_side_imb), 152 基线无 side-split 撤单
├── 事件时距原始聚合 (iat/burst/recent-prev) ❌ P9-B-LITE −0.0039 frozen (RED, 基线已覆盖)
│     → 152 已含 o_*_near_far / t_*_gap / rowcount_near_far; 原始聚合冗余有害
│     → 事件节奏须经 Z 式 market/tx 条件化才有效 (Z 绿灯), 原始形式无增量
└── M55-lite (L1/L2 DWI + trade entropy)  🟡 P9-C-LITE +0.0005 frozen (YELLOW, 进联合)
```

---

→ 实验索引: [experiment-index.md](./experiment-index.md) → 失败墓地: [failed-experiments.md](./failed-experiments.md)
