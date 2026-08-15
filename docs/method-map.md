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
├── Z 创新特征 (SCFI)             ✅ P5-B/D/E, 双 learner 确认
└── 90 特征甜点位                 ✅ C1-FE 证伪更多特征
```

## Metric-aware 指标感知
```
├── cosine loss                    ⚠️ P4-08A INVALID (验证偏差); 生产禁混辅助 loss
├── MSE (标准)                     ✅ 生产默认
├── hybrid MSE+cosine              ❌ 4.14 实锤名义混合实际单任务
├── RMS 尺度校准融合               ✅ C4 冻结规则
└── 全局尺度无关性                 ✅ F001 (浪费全局校准)
```

## Market sequence 市场序列
```
├── market-only 序列               🟡 P5-01 ~+0.0005 (大部分已在聚合特征)
├── 600s 长上下文                  ✅ P4-01a H4 直接证据
├── market history 聚合特征        🟡 P4-MH ~+0.0005
├── mid_range/mid_std 变体         ❌ P7-AMP (对 cosine 无用)
└── TCN 序列模型                   ❌ v6 0.082 灾难 (N005)
```

## Conditional information 条件信息
```
├── market → 状态表示              🟡 E01 ReVol-lite +0.0011 (gate 未过)
├── 状态条件化拼接                 ❌ P3-02 (稀释)
├── 订单流创新 (SCFI: O−E[O|M])   ✅ P5-B/D/E — 当前最有效方向
├── 残差条件化 (Reconditionor)     ❌ E02 (窗口内≠跨月)
├── FiLM / 条件 adapter            🧪 未测试 (完整版, 等 SCFI 系列消化)
└── 事件时条件强度 (ETCI)          🧪 未测试 (plan v1.9.0 队列)
```

## Residual modeling 残差建模
```
├── residual target (y−ŷ)          ❌ P5-03 (AUC 0.51 预判死)
├── OOF 残差均值                   ❌ P4-06A, M01-A (+0.0000)
├── 检索-残差 (P6R FAISS KNN)      ❌ P6R-00 (gate 39%, KILL)
├── 局部回归 (P6R-01)              🧪 挂起 (等拍板)
└── 五连杀总结                     ❌ F011: 可解释≠可预测
```

## Gating / Amplitude 门控与幅度
```
├── 4-bin 幅度 gate                ❌ P5-A (−0.000146 嵌套)
├── γ 幂族调制                     ❌ P5-A (同判)
├── volatility confidence          ❌ P7-AMP (α 单调=0)
├── 时间衰减加权                   ❌ E1-TW
├── 特征相对化                     ❌ P1-01e (第二轮)
└── "Market→Confidence" 假设       ❌ P7-AMP 证伪 (GPT1 round2 核心主张)
```

## Unsupervised / 表示学习
```
├── SAE 稀疏编码                   ❌ P3-01
├── TinyLOBERT 掩码预训练          ❌ P3-03 (corr 0.86~0.98)
├── 2.5D 网格投影                  ❌ P3-04
├── NHP/Hawkes 强度                ❌ P3-05 (|r|≤0.01)
├── Path Signature                 ❌ M03
└── SSL 预训练                     🧪 未测试 (plan 第二阶段, 低优先)
```

## 形状 / 谱 (反演不变)
```
├── 短窗跨通道几何 (RICS)          ❌ P5-C (≤0.011)
├── lag-cov / 相位                 ❌ P5-C (R4 -0.006 反转)
├── SPCE 四臂 probe                🧪 未测试 (plan v1.9.0 队列, 信号被 P5-C 质疑)
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
├── 资产身份 (H1)                  ❌ P4-H1H2
├── 时间身份 (H2)                  ❌ P4-H1H2
├── 600s 流 (H4)                   ✅ P4-01a
├── 未知 factors (H4)              🟡 P4-02 (OFI 弱正)
├── target 逆向 (H3)               🟡 P4-03 未唯一确定
├── 月度漂移预测 (H3)              ❌ P4-05
└── LB142 分歧样本                 🟡 P4-LB142 (高波动集中)
```

## 未测试方向 (plan v1.9.0 剩余)
```
├── O→T lag response               🧪 GPT P1.5 — 当前唯一推荐 (152 无覆盖)
├── SPCE 四臂 / TRIS random        🧪 低优先 (P5-C 已弱化谱线)
├── hard regime experts            🧪 低优先 (幅度路线已死, 需重新论证)
├── soft MoE                       🧪 依赖 P6-04
├── ETCI 事件时臂                  🧪 中等 (与 lag response 同源)
└── P6R-01 终裁                    🧪 挂起等拍板
```

---

→ 实验索引: [experiment-index.md](./experiment-index.md) → 失败墓地: [failed-experiments.md](./failed-experiments.md)
