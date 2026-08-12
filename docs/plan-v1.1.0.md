# MSCapital 科研化方案 v1.1.0

> 来源: GPT1 + GPT2 双评审(2026-08-11)合并 × Hermes 实测核验
> 状态: **评审吸收稿, 待用户拍板后执行**
> 前版: v1.0.0 (2026-08-10)

## 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0.0 | 2026-08-10 | 初版: 吸收 GPT 科研方案核心纪律 |
| v1.1.0 | 2026-08-11 | 合并双 GPT 评审: 验证体系重构 (P0), 对抗验证 (P0), 微观结构表示 (P1), 轻量序列 (P1), 修正 2 个过早结论 |

## 双 GPT 评审共识 (均已吸收)

两位 GPT 独立评审, 结论高度一致:

1. **P0 验证体系**: CV1 只证明"绝对数值接近 LB", 未证明"模型排序与 LB 一致" → 需 Temporal Generalization Matrix (多折叠: T1 m0-30/31-40, T2 m0-40/41-50, T3 m0-50/51-60, T4 m0-50/61-70, H1-H3 跳远折叠) + **Pseudo-LB38** (train m0-32 / valid m33-70, 38个月模拟 test 71-108) + 月度 cos 衰减曲线
2. **P0 Adversarial Validation**: train vs test domain classifier (LightGBM), AUC + 特征重要性 → 找出漂移特征 (AUC>0.8 说明 CV1 未模拟 test; 高漂移特征 = 可疑特征)
3. **P1 微观结构表示** (非堆特征): OFI/MLOFI(1907.06230 已核验)/signed flow/大单不平衡/arrival-cancel intensity/queue depletion-refill/多尺度 1-60s; 10-30 个与现有 90 维低相关的新变量
4. **P1 轻量序列**: 先 CNN/TCN/GRU (DeepLOB 路线), **不直接上 Transformer**; 序列 tensor 建议 [60s × 16ch] + market 5s/10s bars [60-120 steps]
5. **P2 sequence+tabular 融合** (z1/z2/z3 → predict)
6. **P3 metric-aware OOF 融合**: cos 目标 → OLS/Ridge/NNLS 约束权重 (w≥0, Σw=1), 跨 fold 权重稳定性验证 (替代普通 stacking, 防学习 CV regime)
7. **新模型晋级标准** (替代 CV1+0.001): ①CV1 提升 ②rolling-CV 平均提升 ③最差 future fold 不降 ④衰减斜率更小 ⑤与当前 ensemble 相关 <0.90 (③④⑤ > ①)

## 修正: 我方 2 个过早结论 (GPT2 指出, 证据支持)

| 原结论 | 修正为 | 理由 |
|---|---|---|
| "CV1 最诚实" | "CV1 绝对数值最接近 LB (0.130→0.117, 0.139→0.122); **排序一致性未验证**" | 我们只有 2 个提交点, 不足以证明排序一致; 需 Temporal Matrix |
| "表格融合路线 LB≈0.122 封顶" | "基于现有 90 维统计特征的同质模型扩展边际递减" | LB 显示精度 0.001, v1/v2 可能实际不同 (0.1215-0.1224); 换表示/换模型类型仍有空间 |

## 修订路线图 (v1.1.0)

```
P0-1 Temporal Robustness Benchmark   (19_temporal_matrix.py)
     T1-T4 + H1-H3 折叠矩阵, 月度cos曲线, 衰减斜率
P0-2 Adversarial Validation          (20_adversarial_validation.py)
     train vs test classifier, AUC, 漂移特征TOP20, 特征组漂移表
P0-3 漂移特征消融                     (21_drift_ablation.py)
     删除高漂移特征 vs baseline90, CV+矩阵双评估
P1-1 微观结构表示 v1                  (22_microstructure_repr.py)
     OFI/MLOFI/signed flow/大单/强度/多尺度 (10-30新变量, 低相关)
P1-2 轻量序列模型                     (23_seq_model.py)
     1D CNN / TCN / GRU on [60×16] 序列 + market bars
P2   序列+表格融合                     (24_seq_tabular_blend.py)
     z1+z2+z3 结构, cos 优化权重
P3   metric-aware OOF 融合优化        (25_oof_cosine_blend.py)
     OLS/Ridge/NNLS, 跨fold权重稳定性
```

每步 Go/No-Go: 晋级标准 5 条 (见上), 不满足即停/回退。

## 待用户拍板

1. 是否按 v1.1.0 路线执行? (P0-1+P0-2 先行)
2. 本地执行 vs Kaggle GPU: P0 全本地 (快); P1-2 序列模型本地 8GB 先试小配置, 大配置 Kaggle
3. 报告修正: 同意将 "饱和" 改为 "边际递减" (已并入 v1.1.0)
