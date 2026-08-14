# MSCapital 科研化方案 v1.0.0

> 来源: GPT 科研方案(2026-08-10 转述) × Hermes 实测评审合并
> 状态: **评审稿, 待用户拍板后执行**
> 项目: D:\mscapital-kaggle (代码) + D:\mscapital-forecasting (数据)

## 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0.0 | 2026-08-10 | 初版: 吸收 GPT 科研方案核心纪律, 修正 3 处, 拒绝 1 处, 与侦察情报合并 |

## 评审结论 (Hermes 对 GPT 方案的实测评估)

### 采纳 (科研纪律骨架)

1. Evidence before complexity + One-Question-One-Experiment
2. Baseline Ladder B0→B9 (B1=官方基线已就位)
3. Validation Protocol 冻结版本 (防挑选有利CV)
4. 负面结果记录 + Decision Log
5. 防 AI 伪科研: 论文状态标签 Found→Metadata→Abstract→Full→Methods→Result→Relevant
6. 双轨指标: Research Metric (walk-forward OOF/recent CV/regime) + Competition Metric (LB)
7. 文献→RQ→Hypothesis→Experiment 流程 (瘦身版)

### 修正 (证据见下)

| # | GPT 原案 | 问题证据 | 修正 |
|---|---|---|---|
| 1 | 禁止一上来 LightGBM | 自相矛盾: Ladder B1 即官方基线, 而官方基线 lgb_baseline(0.117) 就是 LightGBM, 代码已在 reference/ | B1 先跑通作锚点 (固定官方参数不调参) |
| 2 | 30核心+50~100扩展+每篇20字段Card | 2个月赛程+学业, 文献会拖死比赛 | 核心10~15篇 Deep Read + 30篇轻量Card; 文献时间盒3-4天 |
| 3 | 通用模板 RQ | 已有侦察: CV/LB脱节(Transformer CV 0.1549/LB 0.120)、融合+0.02、全-1=+0.007、90特征甜点位 | RQ 从真实观察出发 (见 RQ 列表) |

### 拒绝

- "禁止 Leaderboard probing": 过绝对。诊断性 probing (sanity check) 有科研价值, 每天5次配额足够; 禁的是盲目 probing。

### skill 盘点 (TASK 001 实测结果)

有: arxiv / primary-source-research / karpathy-wiki-method / fact-check-social-media / blogwatcher
无 (不假装存在): research-paper-writing / literature review 专用 skill → 用 arxiv + primary-source-research + karpathy-wiki-method 组合替代

## v1 方案: 阶段划分

### Gate 0 — 侦察 (已完成 2026-08-10)

- 比赛档案/数据schema/规则/榜单/讨论帖/公开notebook 全部摸清, 见 README.md 情报速查
- 官方基线代码 (lgb_baseline 92特征, LB 0.117) + RealMLP/Transformer基线 已入库 reference/

### Gate 1 — 文献 + 研究设计 (时间盒: 3~4天, 与B1并行)

1. B1 锚点: 跑通官方基线 (数据就绪即跑, 固定参数) → 记录 B1 CV/LB
2. 文献: 核心 10~15 篇 Deep Read (Paper Card 完整字段) + 30 篇轻量 Card
   - 主题优先级: ①OFI/order flow (Cont, Gould 等) ②LOB 深度学习 (DeepLOB 系) ③temporal generalization/regime shift ④Kaggle LOB 赛复盘 (Optiver 两届, Tier B 但实战价值高)
   - 工具: arxiv + primary-source-research; 防伪科研标签全程生效
3. 产出: research_landscape.md / research_gaps.md / RQ + Hypotheses (v1) / validation_protocol_v1.md (冻结)
4. 交付: Research Gate 1 汇报 (10问) → 用户确认后才进正式实验

### Gate 2 — 实验 (One-Question-One-Experiment)

- Family A: 可靠 CV 能否建立 (month-split 敏感性/embargo/分布漂移诊断, 对应 RQ1)
- Family B: 哪些 microstructure 特征真正有效 (特征证据表驱动, 对应 RQ2/RQ3)
- Family C: 跨时间稳定性 (regime 切分评估, 对应 RQ6/RQ7)
- 之后才进入: 序列模型 (RQ4) → 互补性分析+融合 (RQ5)

### Gate 3 — Kaggle 冲刺 (最后2~3周)

- 多模型融合/stacking, 诊断性 probing 校准, 2个最终提交
- Research Metric 与 LB 双轨记录, 不因 LB 破坏实验纪律

## Research Questions v0 (从侦察情报出发, Gate 1 后定稿)

- RQ1: 为什么离线 CV 排名与 LB 排名不一致? (Transformer CV 0.1549 但 LB 0.120) — 分布漂移的量化与应对
- RQ2: 90特征甜点位背后的机理: 特征增多为何 CV 涨 LB 跌? (过拟合 vs 漂移?)
- RQ3: OFI/microstructure 特征的预测力跨 regime 是否稳定?
- RQ4: 树模型与序列模型的错误相关性是否低 (融合 +0.02 的来源)?
- RQ5: 诊断性常数预测 (+0.007) 反映的 target 结构如何利用?

## 目录结构 (合并既有, 不另起炉灶)

```
D:\mscapital-kaggle\
├── docs\            ← 本方案文档 + research_landscape.md 等 (新增)
├── research\        ← 文献库: papers/ paper_cards/ feature_evidence.csv 等 (新增, 待Gate1)
├── scripts\         ← 已有: 01_build_features.py / 02_train_lgb.py (待写)
├── notebooks\       ← 已有
├── output\submissions\  ← 已有
├── RESULTS.md       ← 实验记录 (待建)
└── README.md        ← 已有
D:\mscapital-forecasting\data\raw\ ← 解压中; reference\ 已有
```

## 待用户拍板

1. 科研模式 vs 务实模式: 完全按 Gate1 文献流程 (3-4天) vs 文献压缩到1-2天直接实验?
2. B1 官方基线是否立即跑 (数据解压完成后)?
3. 方案文档确认后即定稿, 版本号升 v1.0.1?
