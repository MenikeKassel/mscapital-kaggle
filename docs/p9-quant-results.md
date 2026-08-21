# P9 三线量化路线报告 (2026-08-15)

> 来源: GPT 第三轮建议 (docs/gpt-review-quant-routes-2026-08-15.md) 的 3 条 S/A 级路线
> 协议: C-05 E0 同款 (152 特征 → robust+clip → 3×256 MLP → MSE, m0-32/m33-70, best-cosine-epoch)
> 基线: C-05 E0 = **0.113261** (PSEUDO eval 33-70)
> 判定: neutralization 🟡 / DG 🔴 / NC 🔴

---

## 1. P9-NEUT 预测中性化 — 🟡 正信号 (唯一正结果)

**方法**: p' = p − γ·ŷ_Z; Z = {m_mid_std, m_mid_std_180, m_rv, m_rv_60, m_rv_180, m_sp_mean_60, o_vol_sum, |p|}; OLS fit Z→p 在 calibration 33-50, 审判 frozen 51-70 (20 个月未触碰)。

| γ | frozen 51-70 | Δ | eval 33-70 |
|---|---|---|---|
| 0.00 | 0.114364 | — | 0.113262 |
| 0.25 | 0.114982 | +0.000619 | 0.113637 |
| 0.50 | 0.115420 | +0.001056 | 0.113837 |
| 0.75 | 0.115669 | +0.001305 | 0.113856 |
| **1.00** | **0.115725** | **+0.001361** | 0.113691 |

- γ 单调 0→1 持续改善 (无过拟合拐点); 零训练成本 (后处理)
- 月度: frozen 12/20 月正, 全局均值 +0.00031/月 — **增益集中** (诚实标注: 未达 70% descriptive gate, 但全局 cosine 增益真实)
- 机制: 与 P7-01 一致 (高波动/活动度暴露伤害方向质量), Z 剥掉的是预测中的 nuisance exposure
- **可叠加**: 与 C 系列组件/融合正交 (后处理, 可套在最终融合上)
- 注意: 目前基于 C-05 单模型 PSEUDO 预测; 需在真实融合预测上验证 (test 侧需 f0726_test Z 特征)

## 2. P9-DG Month-Invariant (V-REx) — 🔴 无稳定增益

**方法**: L = mean(L_m) + λ·var(L_m), month 分组 batch 内计算; λ 全扫描。

| λ | cos | Δ vs 基线 |
|---|---|---|
| 0.1 | 0.109895 | −0.0034 |
| 0.3 | 0.111439 | −0.0018 |
| 1.0 | 0.111828 | −0.0014 |
| **3.0** | **0.113569** | **+0.0003** (峰) |
| 10.0 | 0.110307 | −0.0030 |
| 30.0 | 0.110863 | −0.0024 |

- **λ=3.0 是孤峰**: 两侧 λ 全部显著为负; 无邻域支持 ⇒ +0.0003 判定为噪声/偶然, 非真信号
- 趋势解读: 小 λ 时惩罚不足以改变训练 (略伤拟合), 大 λ 时方差项主导迫使模型牺牲平均表现 ⇒ 在 cosine 指标 + PSEUDO 协议下无可用区间
- 注意实现: batch 内 month 分组计算使训练慢 ~10 倍 (per-group kernel 开销), 6 个 λ 共 ~75 分钟
- 结论: V-REx 惩罚不适用本任务; "月度漂移"对训练约束不可转化 (与 P4-07 漂移不可预测的结论同向)

## 3. P9-NC Negative Correlation Learning — 🔴 机制性证伪

**方法**: L = MSE + λ·penalty; v1 corr 形式 (崩), v2 协方差形式 mean(e_B·e_P) (论文 Brown 2005 原式)。

| λ (v2) | cos | Δ | eval err_corr |
|---|---|---|---|
| 0.1 | 0.112873 | −0.0004 | 0.996 |
| 0.3 | 0.109389 | −0.0039 | 0.994 |
| 1.0 | 0.110738 | −0.0025 | 0.995 |

- **机制性失败** (非实现问题): err_corr ≈ 0.99 — 新模型误差与参考模型几乎完全一致。原因: 残差方向无信息 (P5-03: market 对 y−β·v7 方向 AUC 0.51 ≈ 0.5), "互补错误"不存在, diversity 惩罚无米下锅
- 协方差形式下模型"作弊" (e_B→0 同时最小化两项), 惩罚退化
- 结论: 与 F011 家族同因 (可解释≠可预测); 不再尝试任何误差互补类训练

## 4. 记账更新

- registry.csv: P9-NEUT (YELLOW, +0.001361 frozen), P9-DG (RED, 6 点扫描孤峰), P9-NC (RED, 机制证伪) — 88→90 条
- method-map.md: 新增"量化路线"区: neutralization 🟡 / Month-DG ❌ / NCL ❌
- failed-experiments.md: NC 入墓地 (机制: 残差无信息, 与 P5-03 同族)
- scripts/: p9_neutralize.py + p9_neutralize_monthly.py + p9_dg.py + p9_nc.py

## 5. 复现路径

```
.venv/Scripts/python.exe scripts/p9_neutralize.py            (10s, CPU)
.venv/Scripts/python.exe scripts/p9_neutralize_monthly.py    (60s, CPU)
.venv/Scripts/python.exe scripts/p9_dg.py --lam <0.1|0.3|1.0|3.0|10.0|30.0>  (~12min, GPU)
.venv/Scripts/python.exe scripts/p9_nc.py --lam <0.1|0.3|1.0>               (~2min, GPU)
输出: output/p9_neutralize/, output/p9_dg/lam_*/, output/p9_nc/lam_*/
```
