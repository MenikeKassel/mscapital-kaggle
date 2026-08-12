# 本地指标 vs 提交分数 校准 (v3, 2026-08-12, 吸收 GPT1 评审 + Regime B 定标)

## 提交记录校准表

| 提交 | 内容 | PSEUDO-LB38 (本地) | LB (真实) | 差值 (PSEUDO-LB) | Regime |
|---|---|---|---|---|---|
| v1 | LGBM+XGB+MLP (CV1权重) | 0.12916 | 0.122 | +0.00716 | A 传统 tabular |
| v2 | XGB+Cat+MLP (重复观测) | 0.12916 | 0.122 | +0.00716 | A |
| v3 | temporal权重 | 0.13002 | 0.122 | +0.00802 | A |
| v4 | R2 归一化 | 0.13132 | 0.123 | +0.00832 | A |
| v5 | R2+22微观 | 0.13487 | 0.125 | +0.00987 | A |
| v6 | +TCN 0.07 | 0.13792 | 0.082 ❌ | 序列退化 (Gate 失败) | 高风险类别 |
| v7 | +RealMLP 0.2 | **0.13968** | 0.135 | **+0.00468** | B 新 representation |
| v8a | +lb142ref 0.3 | 无共同验证 | 0.139 | - | C external probe |
| v8b | +lb142ref 0.5 | 无共同验证 | 0.142 | - | C external probe |

## Regime A (传统 tabular, v1-v5): LB ≈ PSEUDO − 0.008

差值: 0.00716/0.00716/0.00802/0.00832/0.00987, 均值 0.00811
- v1/v2 是近重复观测 → 有效独立样本 < 5, **均值不写得太精确**
- 观察到高 PSEUDO 区间 gap 有扩大迹象 (0.129→0.007, 0.135→0.010), 但样本不足, **暂不建模为确定趋势** (两种解释: calibration compression vs 特征对 PSEUDO 更有利, 未区分)
- 强模型阶段保守按 **−0.009~−0.010** 估计

## Regime B (新 representation / RealMLP, v7 起)

首个定标点已完成（Kaggle v12，m0-32 / m33-70，最佳 EMA）:
- RealMLP 单模型 PSEUDO = **0.138560**
- v5 表格 PSEUDO = **0.134871**
- 实际 v7 原始权重 `0.8 table + 0.2 RealMLP` = **0.139683**（相对表格 +0.004813）
- v7 LB = 0.135 → gap = **0.004683**，明显小于 Regime A 的 0.008~0.010

但这个点不能直接推广成新的线性校准公式：
- 只有 1 个独立 Regime B 观测，且 `w=0.2` 曾由 Public LB 比较确认，存在选择偏差。
- 两模型原始尺度差异很大；valid std 为 table 0.000341 / RealMLP 0.015322，权重不是等尺度权重。
- test 上 RealMLP std 降至 0.011159（test/valid=0.728），导致 v7 融合 test/valid std=**0.745**。
- valid/test corr(table, RealMLP) = **0.8658 / 0.8263**；相关结构尚稳定，但幅度发生明显迁移。

**解释**: Regime B 的真实 LB 转化优于 Regime A，但预测尺度迁移更强。后续 Dynamics/RealMLP 实验必须同时报告 raw blend 与 component-wise 归一化后的诊断；不得把 `LB ≈ PSEUDO − 0.0047` 当成通用公式。

## Regime C (external prediction, v8 起)

无共同 validation → **不进入 PSEUDO→LB 校准模型**, 只作为独立 ensemble probe 管理。
- v8 权重扫描是 Public LB probe (已用 0.3/0.5), **不继续机械扫 0.6/0.7/0.8** (拿 Public LB 当 validation 的风险)
- v8b 正确含义: 外部模型含我们没有的 alpha (corr 0.82 非相关信息 18%), 应**反向研究 ref 为什么有效**

## 两层校准体系

### Layer 1: Score Calibration (分 regime 记录)
model family / features / PSEUDO / LB / gap — Tabular、Dynamics、Sequence 分开, 不混成一个公式

### Layer 2: Prediction Calibration (每个提交必记)
```
valid std / test std / test-valid std ratio
mean, p01, p05, p50, p95, p99
corr_v5, corr_v7, corr_best
```
v6 教训: PSEUDO 0.1379 看似能冲 0.13, 实际 test/valid std 异常 → 灾难。**分布/尺度门禁比分数校准更重要**

## 提交门禁 (制度化)

1. **PSEUDO/temporal folds 稳定** (目标 0.145 → PSEUDO ≥ 0.155, 无安全边际不提交)
2. **test pred 分布**: mean/std/min/max/分位数正常
3. **Scale**: std_test/std_valid 不严重偏离 (异常 → 禁止)
4. **Correlation**: 与 v5/v7/最佳提交对比记录
5. **Blend sanity**: 融合后 std 不突变、mean 不漂移、无极端值

> **制度**: TCN 路线冻结; 任何新模型必须同时通过 PSEUDO、test distribution、prediction scale、correlation 门禁; 外部预测不进入本地 score calibration, 只作为独立 ensemble probe 管理。

## 待补实验
- [x] RealMLP PSEUDO (Kaggle v12) → v7 gap 定标 (Regime B 首个点；详见 `output/rlps_v12/v7_pseudo_diagnostics.json`)
- [ ] ref forensic: delta = ref_pred − v7_pred, 按 volatility/spread/depth/event intensity/near-far/pred magnitude 分桶, 找 disagreement 的 regime 结构
- [ ] ΔPSEUDO per feature family 指标 (Baseline + dynamics V2 +0.00XX 逐族记录)
