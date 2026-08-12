# MSCapital 科研化方案 v1.1.1

> 来源: GPT1 + GPT2 二轮评审拍板 (2026-08-11) × Hermes
> 状态: **已批准, 执行中 (P0-1/P0-2)**
> 前版: v1.1.0

## 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0.0 | 2026-08-10 | 初版 |
| v1.1.0 | 2026-08-11 | 双 GPT 评审: 验证体系重构/对抗验证/微观结构表示/轻量序列 |
| v1.1.1 | 2026-08-11 | GPT 二轮拍板: GO; 晋级规则细化 (硬门槛③④, ⑤=稳定正增益); Pseudo-LB38 独立; Adversarial 三组; 冻结模型扩展与提交直至 P0 完成 |

## 执行令 (GPT1+GPT2 双批准)

1. **冻结**: P0 完成前不训练新模型类型、不堆特征、不提交 Kaggle (剩余提交配额留作科研验证)
2. **P0-1 Temporal Matrix 先行**: 固定 LGBM/XGB/CatBoost/MLP-ens/Blend 参数, 不调参; 折叠: T1(m0-30/31-40), T2(m0-40/41-50), T3(m0-50/51-60), T4(m0-50/61-70), H1(m0-30/41-50), H2(m0-40/51-60); 输出 fold×model 矩阵, mean/std/worst, Δ 相对基线, decay slope, **Alpha half-life** T1/2=ln2/λ, 排序 Spearman 稳定性
3. **Pseudo-LB38 独立**: train m0-32 / valid m33-70 (38月模拟 test), 作为 simulated hidden test, 不进 rolling 均值
4. **P0-2 Adversarial Validation**: 三组 (m0-50 vs m51-70 / m51-70 vs test / full train vs test), AUC + 特征重要性 + 特征组 drift 表 + Predictive×Drift 四象限
5. **晋级规则 v2** (替代 CV1+0.001):
   - Hard gates: ③最差 future fold 不降 ④decay 斜率不恶化
   - Strong preference: ②rolling mean 提升 ⑤加入 ensemble 后多 fold 稳定正增益 (corr<0.90 仅前置诊断)
   - Weak evidence: ①CV1 提升 (CV1 不得灾难性退化, 不要求必升)
6. **P0 后三叉分叉**: A 共同衰减→invariant representation; B 排序翻转→robust model selection 先行; C 某 family 半衰期长→围绕其扩展 (如 order-flow → OFI/MLOFI 重点)

## 产出物 (P0 完成时)

- RESULTS.md 追加: 折叠矩阵表 + half-life 表 + adversarial 三组 AUC + 特征四象限
- 模型报告格式: CV1 / mean / worst / half-life / corr
