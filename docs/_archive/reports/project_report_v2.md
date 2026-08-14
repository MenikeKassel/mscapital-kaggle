# MSCapital 项目阶段报告 v2 (P0 完成, 供 Codex 评审)

> 日期: 2026-08-11 02:50 | 前版: project_report_v1.md (2026-08-11 00:55)
> 本轮进展: 执行了双 GPT 评审批准的 P0 阶段 (Temporal Matrix + Adversarial Validation + 权重重估), 完成 3 次提交, 确认平台期, 提出 P1 方案。

## 1. 上轮评审后执行情况

双 GPT 评审 (v1.1.0/v1.1.1) 批准: 冻结模型扩展, P0 先行。已执行:

| 任务 | 状态 |
|---|---|
| P0-1 Temporal Generalization Matrix (7 folds × 5 模型) | ✅ |
| P0-2 Adversarial Validation (三组) | ✅ |
| P0-3 融合权重重估 (temporal 优化) | ✅ |
| v3 科研验证提交 (temporal 权重) | ✅ LB 0.122 |
| 提交配额: 今天已用 3 次, 剩 2 次 | |

## 2. P0-2 Adversarial Validation 结果

| 组 | 对比 | AUC | 解读 |
|---|---|---|---|
| A | m0-50 vs m51-70 | 0.7334 | train 内部漂移已明显 (持续演化) |
| B | m51-70 vs test | 0.7772 | 近期 vs test 漂移最大 |
| C | full train vs test | 0.7659 | 整体漂移确认 |

**机制发现: 预测主力 = 漂移主力**
- 最大漂移特征: m_sp_mean (价差均值, 三组 TOP1, C 组 gain 是第 2 名 1.4 倍), m_depth_mean, m_rv, o_vol_sum/o_n_120
- 特征组漂移 (C): m_book 45.1% > o_order 21.8% > m_window 18.6% > t_transaction 10.6% > x_cross 2.4% > m_ewm 1.5%
- **解释**: CV-LB 差 0.017 来自价差/深度/波动特征在 train→test 系统性漂移; E1 时间加权无效是因为漂移是市场状态演化而非简单时间距离

## 3. P0-1 Temporal Matrix 结果

折叠矩阵 (cos, 全部固定参数不调参):

| fold (valid) | lgb | xgb | cat | mlp | blend |
|---|---|---|---|---|---|
| T1 (m31-40) | 0.11826 | 0.11558 | 0.11756 | 0.11062 | 0.12047 |
| T2 (m41-50) | 0.12360 | 0.12055 | 0.12604 | 0.11611 | 0.12726 |
| T3 (m51-60) | 0.12714 | 0.12603 | 0.12760 | 0.11961 | 0.13000 |
| T4 (m61-70) | 0.13395 | 0.13539 | 0.13850 | 0.13346 | 0.14517 |
| H1 (m41-50, 跳远) | 0.11903 | 0.11902 | 0.12172 | 0.11414 | 0.12428 |
| H2 (m51-60, 跳远) | 0.12743 | 0.12518 | 0.12705 | 0.11821 | 0.12883 |
| PSEUDO-LB38 (m33-70) | 0.12423 | 0.12230 | 0.12482 | 0.11963 | 0.12916 |

**三大发现**:
1. **CV1 模型选择能力证伪 (GPT1 预言命中)**: MLP 在 CV1 第一 (0.1337) 但在全部 7 个 temporal folds 垫底 (mean 0.1187)。MLP 的 CV1 优势是 m61-70 近期 regime 特有。
2. **CatBoost 是最稳健单模型** (temporal mean 0.1264, 5/7 folds 单模型第一); blend 每个 fold 全胜 (mean 0.1293, +0.003)。
3. **half-life 不适用**: 月度 cos 随月份上升 (61-70 > 51-60), λ<0。test 期任务可预测性可能更高 (可能高波动 regime), CV-LB 差更多来自训练分布距离。
4. 排序 Spearman: T1/T4 与其余 fold 相关 0.70, 其余 0.90+ (大体稳定, 有边界翻转)。

## 4. P0-3 权重重估 + 三次提交

- temporal-mean 最优权重: (lgb 0.2, xgb 0.0, cat 0.5, mlp 0.3) — MLP 降权 (0.5→0.3), CatBoost 升权 (0.4→0.5)
- Pseudo-LB38 验证: 旧 0.129155 → 新 0.130015 (+0.0009)
- **三次提交全部 LB 0.122**: v1 (CV1权重), v2 (+CatBoost), v3 (temporal 权重)
  - LB 显示精度 0.001, 三次实际可能在 0.1215-0.1224 区间内
  - **结论: 表格融合权重微调无法突破, 0.122 是真实平台期 (信息源瓶颈, 非权重/模型调优问题)**

## 5. P0 决策门判断

- 三叉分叉 = 情况 A+B 混合: 排序大体稳定 (Cat 强 MLP 弱) 但 CV1 会给出翻转排序 → **robust model selection 体系已建立** (temporal mean + Pseudo-LB38 + worst fold + 衰减斜率)
- **核心结论: 突破需要结构性变化 (P1), 不是继续调表格模型**

## 6. P1 方案 (待 Codex 评审后执行)

| 优先级 | 实验 | 内容 | 执行地 |
|---|---|---|---|
| P1-1 | 微观结构表示 | OFI/MLOFI/signed flow/大单不平衡/arrival-cancel 强度/queue 变化/多尺度 1-60s; 目标 10-30 个与现有 90 维**低相关**新变量 (换表示非堆特征) | 本地 |
| P1-2 | 轻量序列模型 | [60s×16ch] 序列 tensor (订单/成交秒级聚合) + market 5s/10s bars; CNN/TCN/GRU 先行, 非 Transformer | Kaggle GPU |
| P2 | 序列+表格融合 | z1 (序列) + z2 (表格) → 预测; 晋级标准: 多 fold 稳定正增益 + corr<0.90 | 本地+Kaggle |

**晋级规则 v2 (已生效)**: Hard gates = 最差 future fold 不降 + 衰减不恶化; Strong = rolling mean 提升 + ensemble 稳定正增益; Weak = CV1 提升 (不得灾难性退化)。

## 7. 给 Codex 的评审问题

1. P0 结论是否有漏洞? 特别是: ①"MLP 是假冠军"是否受 MLP 训练配置 (30ep, lr 1e-3) 影响 (更多训练能否让 MLP 在短训练 folds 上翻身)? ②月度 cos 上升的另一种解释 (非可预测性, 而是 target 分布随月份变化)?
2. P1-1 微观结构表示的具体特征清单建议? 哪些与现有 90 维 (窗口统计/EWM/签名量) 相关性最低?
3. P1-2 序列模型的输入设计: 秒级聚合 vs 事件流; 归一化方式 (价差/波动归一 vs 原始值) 对漂移稳健性的影响?
4. 表格平台期 0.122 的判断是否成立? 有无遗漏的表格侧杠杆 (如 per-month 分模型/regime 切换模型)?
5. Kaggle GPU 工作流: 特征 parquet 上传 vs Kaggle 上重建, 哪个更稳?

## 8. 资产状态

```
提交: v1/v2/v3 全部 0.122 (今天剩 2 次配额)
实验结果: RESULTS.md 完整 (17+7 实验)
脚本: scripts/00-22 (全可复现)
方案: docs/plan-v1.1.1.md
数据: D:\mscapital-forecasting (raw + processed parquet)
环境: 本地 RTX 4060 Ti 8GB / 32GB RAM; Kaggle GPU 30h 未用
```
