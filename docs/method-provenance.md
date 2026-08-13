# 方法溯源库 (Method Provenance)

> 来源: ChatGPT 分享页 × Hermes 交叉核实 (2026-08-13)
> 用途: 每条候选方法标注真实来源 (Origin), 防止"给比赛启发式找不存在的原论文"和"把内部假设写成冠军方法"
> 与 `method-transfer-sprint.md` 的 M01-M06 对齐

## 战略结论 (先记住这个)

> MSCapital P2 的目标不是寻找"比 RealMLP 更好的模型", 而是寻找"比当前 152 dynamics 更好的市场表示"。

"表示 > 架构"不是猜测, 有 4 条独立研究线共同支撑 (Deep OFI / JPM robust representation / MLPLOB / TabPFN-TS), 且与本项目自身证据同构:
- N003: Transformer 深度优化 CV 0.1549 → LB 0.120
- N005: TCN 融合 PSEUDO +0.004 → LB 0.082 (test corr 坍缩到 0.03)
- v7: 152 dynamics 特征 + RealMLP = +0.010 (最大单步)
- JPM 实验: 同一 MLP 仅换表示 → +11%

## 血缘纠正 (3 条, 已核实)

| 方法 | 纠正 | Origin 标注 |
|---|---|---|
| RealMLP | 不是金融方法; 血缘是 tabular ML research。0.125→0.135 的启示不是"RealMLP 强", 而是"金融数据被重新表达后简单 tabular learner 变强" | `tabular ML (NeurIPS 2024)` |
| Triplet Imbalance | 没有同名原论文; 与 triplet loss 无关; 是 Optiver 选手的 feature heuristic | `Kaggle heuristic (Optiver TATC)` |
| Recent-Regime Specialist | 不是正式方法名; 是"Jane Street adaptive NN + static GBDT"与 AdaRNN 思想抽象出的内部工程假设; 找到的明确 adaptive+static 方案仅 Private 162nd | `internal engineering hypothesis` |

## 溯源表 (Origin → 论文/比赛 → MSCapital 价值)

| 方法 | Origin | 关键论文/比赛 (已验证) | MSCapital 价值 |
|---|---|---|---|
| OFI | 微观结构 | Cont et al., arXiv:1011.6402 | S+; 比简单 bid/ask imbalance 有理论根基 |
| MLOFI | 微观结构 | Xu/Gould/Howison, arXiv:1907.06230 | S+; 多档订单流, M01 最该跑的版本 |
| Deep OFI | OFI+DL | Kolm/Turiel/Westray, Math Finance 2023 (期刊, 非 arXiv) | S+; stationary flow + 简单 ANN > raw LOB 复杂模型 |
| Market-Centered / Depth | JPM AI Research | Wu et al., arXiv:2110.05479 | S+; 仅换表示 MLP +11%; M02 直接祖先 |
| Path Signature | Rough Path 数学 | Chevyrev/Kormilitzin, arXiv:1603.03788 | S+; 事件路径 → 固定维度 → RealMLP/GBDT |
| Triplet Imbalance | Kaggle heuristic | Optiver TATC 公开方案 | S; 便宜, dynamics/price/depth 三元交互 |
| Market Urgency | Kaggle heuristic | Optiver TATC | S; spread × imbalance → spread × OFI/intensity |
| Depth Pressure | Kaggle heuristic | Optiver TATC | S; 与现有 22micro+152dynamics 互补 |
| Synthetic Index | Kaggle heuristic | Optiver TATC | S 但前提严格: MSC 需存在横截面共同因子 |
| Zero-sum Calibration | Kaggle post-proc | Optiver TATC 1st (5.4457→5.4405) | 条件候选; 必须先证明 target 近 zero-sum |
| Market-State KNN | Kaggle | Optiver RV 1st "Nearest Neighbors" | S; 与 RealMLP 低相关 alpha 源 (M04) |
| HLOB | 学术 LOB | Briola et al., arXiv:2405.18938 | S/A+; 偷 LOB shape/interaction, 不复制整个网络 |
| DeepLOB | 学术 LOB | Zhang/Zohren/Roberts, arXiv:1808.03668 | 理论参考; raw sequence 路线风险高 (TCN 教训) |
| MLPLOB | 学术 LOB | TLOB 论文 (Berti et al., arXiv:2502.15757) | S; 等 sequence representation 过 test gate 再跑 |
| TLOB | 学术 LOB | 同上 | A; 价值在 dual-axis 思想, 不优先复刻大模型 |
| Label Horizon | 学术 LOB | 同上 | A; 转 proxy/auxiliary target, 不改官方 target |
| AdaRNN | 时序 DA | Du et al., CIKM 2021, arXiv:2108.04443 | A; 迁移 period matching 思想, 不一定要 RNN |
| Recent-Regime Specialist | 内部假设 | JS RTM Private 162nd + AdaRNN 理论 | A; 低成本试 recent-weighted RealMLP/Cat |
| Robust CV + LGBM | Kaggle | Ubiquant 2nd | 已部分吸收 (Canonical OOF/nested/PSEUDO) |
| Cross-sectional Dynamics | 比赛+文献 | Cross-Impact OFI, arXiv:2112.13213; Optiver index | S; 只对 velocity/OFI/vol/intensity 做 rank/zscore |
| SimLOB | 学术 repr | arXiv:2406.19396 | A+; 未来把 300-600 特征压 latent |
| LOBench | 学术 benchmark | arXiv:2505.02139 | 战略依据: 独立低维 LOB representation + 简单模型 |
| TabM | tabular DL | Gorishniy et al., ICLR 2025, arXiv:2410.24210 | A+; RealMLP 后第二个 tabular NN |
| TabPFN-TS | foundation model | arXiv:2501.02945 (论文声称超 Chronos-Large, 未独立复现) | A; simple features + tabular learner 思想 |
| MultiStream CNN+Transformer (LB142) | 外部参考 | 血缘 ≈ DeepLOB→TransLOB→TLOB (arXiv:2003.00130) | 研究对象; 不是成绩主线 |

## M01-M06 最终定义 (与 sprint 对齐)

| ID | 名字 | Origin | 内容要点 |
|---|---|---|---|
| M01 | Multi-Level Dynamic OFI | Cont→Xu→Kolm | event flow (M01-A 已跑) + quote OFI L1/L2 + OFI/depth + rolling OFI + fast-slow + ΔOFI + velocity |
| M02 | Mid-Price-Centered Geometry | JPM Wu et al. | (price-mid)/spread, cumulative depth, slope, curvature, relative depth; 固定维度 summary 给 RealMLP, 不复制 tensor |
| M03 | Path Signature | Rough Path | price/spread/imbalance/OFI/depth/event time 3-6 维 path, depth 2/3 |
| M04 | Residual Market-State KNN | Optiver RV 1st | 历史月份邻居, 预测 v7 残差; 目标 corr(RealMLP, KNN) < 0.8 |
| M05 | Optiver Interaction Family | TATC heuristics | triplet imbalance, market urgency, depth pressure, OFI×spread, OFI×intensity; 实现成本极低 |
| M06 | Cross-sectional Dynamics | Cross-Impact 文献 + TATC index | 只 rank(OFI/velocity/vol) + market_mean + asset-market; 不做全量 relative (N006 教训) |

## 优先级 (alpha 概率 × 成本 × 与 RealMLP 互补度)

1. MLOFI/Deep OFI (M01) → 2. Market-Centered/Depth (M02) → 3. Path Signature (M03)
→ 4. Dynamic Microstructure 交互 (M05) → 5. Cross-sectional (M06) → 6. Market-State KNN (M04)
→ 7. HLOB shape → 8. TabM → 9. MLPLOB → 10. SimLOB → 11. AdaRNN/recent regime → 12. TLOB/Transformer/MultiStream (冻结)

## 链接验证记录

2026-08-13: 15 篇论文/比赛全部经 arXiv API 或官方来源逐条核实, 编号↔标题↔作者一致:
arXiv: 1011.6402, 1907.06230, 2110.05479, 1603.03788, 1808.03668, 2405.18938, 2502.15757, 2108.04443, 2406.19396, 2505.02139, 2410.24210, 2501.02945, 2003.00130, 2112.13213 (+ RealMLP 2407.04491)
Deep OFI 为期刊论文 (Mathematical Finance 2023, IDEAS/RePEc 链接), 不在 arXiv 批量内。
Kaggle 侧: Optiver TATC 1st/50th、RV 1st、Ubiquant 2nd、JS RTM 162nd write-up 均为用户提供链接, 未逐页抓取 (登录墙), 结论性数字以原 write-up 为准。
