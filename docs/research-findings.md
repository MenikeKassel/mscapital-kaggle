# MSCapital 已确认研究结论 (Research Findings)

> 只记录已经实验支持的**事实**, 不记录实验过程。证据 → 实验 ID。
> 置信度: HIGH (多实验/双窗口确认) / MEDIUM (单实验但协议干净) / LOW (单实验或诊断性)
> 更新: 2026-08-15 (仓库工程化整理 Phase J)

---

## 指标与最优预测

**F001 — 全局 cosine 指标下, 最优预测 = E[y|x] (含幅度)**
证据: breakthrough-top3 定理 1-2 (Cauchy-Schwarz + 重期望); 全局尺度无关 (定理 1.1); 大 N 下 |Y| 集中
置信度: HIGH (数学推导)

**F002 — 分数集中在重尾样本 (top20% ≈ 74.4% 内积)**
证据: P5-02I 内积分解; 定理 3
置信度: HIGH

## 数据与信息面

**F003 — 方向信息微弱, 幅度信息巨大**
证据: P5-02I: sign(y) AUC 0.564 (corr_y ~0.038); rank(y) corr 0.089; |y| corr **0.466** (corr_y ~0.016 与 y 本身无关)
置信度: HIGH

**F004 — 高波动样本方向预测质量更差 (幅度加权必然有害)**
证据: P7-AMP 分桶: 低波动 50% cos=+0.215 → 高波动 5% cos=+0.116; P4-07 分层
置信度: HIGH (P5-A + P7-AMP 双实验一致)

**F005 — 600s market 流含独立于 152 特征的信息 (H4)**
证据: P4-01a (AUC +0.019); P5-01 (~+0.0005); 但大部分已并入聚合特征族
置信度: MEDIUM

**F006 — 0.5 价簇是 ask 全空的计算假象, 非第二 instrument**
证据: P4-04 取证 + dataset-sample-walkthrough (mid=(0+1)/2 钉死)
置信度: HIGH

**F007 — train→test 漂移集中在价差/深度/波动特征; 漂移不可预测**
证据: P0-02 (AUC 0.73-0.78, m_book 45.1%); P4-05 (决定性负结果)
置信度: HIGH

## 模型与验证

**F008 — 归一化 (R2) 表示延长 alpha 寿命 (漂移干预因果链)**
证据: P0.5-C (R2 4/4 folds +0.0023); P0.5-D 融合兑现; **v4 LB 0.123 验证**
置信度: HIGH (真实 LB 验证闭环)

**F009 — 表格 PSEUDO 校准可靠, 序列模型不可信**
证据: v5 表格 PSEUDO 0.1349→LB 0.125 (差 0.010 稳定); TCN PSEUDO +0.004→LB 0.082 灾难
置信度: HIGH

**F010 — 融合是最大杠杆 (异质模型 corr<0.9 融合必增)**
证据: G1-G3 (0.1370→0.1389); v7 (RealMLP 0.8/0.2 +0.010); v8b (lb142 0.5/0.5 追平 0.142)
置信度: HIGH

**F011 — 残差可解释性 ≠ 残差可预测增量 (五连杀)**
证据: E02, P4-06A, P5-02I resid, P5-03, P6R-00 (最大 +0.000588 = gate 39%)
置信度: HIGH

**F012 — 幅度可预测 ≠ 幅度加权有效**
证据: P7-AMP: corr(g,|y|)=0.383 但 gate Δ=0; baseline 幅度分布已隐式最优
置信度: HIGH

## 有效方向 (当前资产)

**F013 — 条件创新 (O−E[O|M]) 在 LGB 与 RealMLP 双 learner 确认有效**
证据: P5-B (LGB ΔC +0.0075, 17/20 月); P5-D (3seed +0.000849 双块正); P5-E (RealMLP +0.0040 / blend +0.0014 双窗口)
置信度: HIGH
注: SmallMLP 无增益 → 增益 learner 依赖 (LGB/RealMLP 系)

**F014 — RealMLP-C 生产候选已过全部门禁 (未提交)**
证据: P6: PSEUDO blendΔ +0.001435 (w=0.75) 与 R61_70 +0.001369 一致; std 比 0.9726; test corr(C,v8b)=0.928 结构吻合
置信度: MEDIUM (双窗口一致但未 LB 验证; 期望 LB 0.1423~0.1428, 增量压缩因 v8b 已含 RealMLP 族)

**F015 — 外部公开方案直接融合是合法且有效的突破路径**
证据: v7 (RealMLP 复刻 +0.010), v8b (lb142 包追平最强公开方案 0.142); 三次最大突破全部来自外部方案驱动
置信度: HIGH

## 科学方法论

**F016 — 嵌套协议能识别假增益**
证据: P5-A (非嵌套 +0.000138 → 嵌套 -0.000146); P6R bootstrap CI 门禁
置信度: HIGH

**F017 — 评估必须 best-state; 消融必须严格 nested**
证据: P0.5-B (last-epoch bug); P4-08A (INVALID)
置信度: HIGH

---

→ 证据链入口: [experiment-index.md](./experiment-index.md) → [experiments/registry.csv](../experiments/registry.csv)
