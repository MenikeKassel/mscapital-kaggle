# MSCapital P5-B — SCFI Tabular Conditional Innovation Probe 报告 (2026-08-14)

> 实验: `scripts/p5b_scfi.py` + `scripts/p5b_build_features.py` | 产物: `output/p5b_scfi/{results.json, p5b_preds.npz, raw_ot_agg.parquet}`
> 任务书: "P5-B SCFI Four-Arm Conditional Innovation Probe" (2026-08-14, §5)
> 判定: **SCFI CONTINUE** (arm delta 强正, late 稳定; LIVE 条未达 → 条件模型升级仍被门禁)

## 12.1 Hypothesis

事件流的价值在"给定市场状态下的 surprise" Z = (observed − E[observed|M])/robust_scale,
而非 absolute level。条件化表示应比 raw 更稳定、更可用。
—— **支持**: 五臂中 innovation 臂 (C) 最强, 且 D−E (超出 capacity control) 双块为正。

## 12.2 Data

| 项 | 值 |
|---|---|
| 原始事件流 | order.feather (170M 行, side∈{0,1}, order_action∈{0,1}: 0=add, 1=cancel — 项目 27_build 同约定), transaction.feather (104M 行) |
| 新特征 (本实验构建) | **73 个** raw O/T 聚合: side×action 2×2 拆分计数 (bid/ask add/cancel), signed/act vol, OFI, IAT mean/std/max/CV/burst, size p25/p75/p90, price std, recent-15s-vs-prev-30s 强度比, 大单局部 q90 不平衡 — f0726 缺失的 side 拆分与 burstiness 是新信息面 |
| 状态 M | m05 ReVol-lite market_state 16 特征 (E02 已验证) |
| 下游 | LightGBM (B1 官方参数, lr 0.02/leaves 32/l2 5.0, 早停 200 on temporal holdout) |
| 切分 | B51_60: train 0-48, holdout 49-50, eval 51-60; B61_70: train 0-58, holdout 59-60, eval 61-70 |
| 基线锚 | canonical clean-baseline-v2 OOF (双锚报告: 同 learner Arm A + canonical blend) |

## 12.3 Leakage audit

- [x] Z 全部 cross-fit: train 内 10 月 fold (fit on other blocks → predict block); holdout 用 train-only fit; eval 用全 train fit
- [x] Z 的 scale (MAD) **只来自 train 折外残差** — eval 残差不参与任何尺度计算
- [x] LGB 早停只在 holdout (49-50/59-60, 严格早于 eval); blend 权重只在 holdout 调
- [x] raw 聚合特征只依赖 order/tx 流 + seconds (无 target 参与)
- [x] f0726 自带 ~2% NaN 由 LightGBM 原生处理, 五臂一致
- [x] 任何模型选择未看过 eval 结果

## 12.4 Baselines

| 基线 | cos (B1/B2) |
|---|---|
| Arm A (LGB on 152, 同 learner 对照) | 0.12792 / 0.13406 |
| canonical baseline OOF (生产锚) | 0.14352 / 0.15476 |
| Arm E (A + raw + z², capacity control) | 0.13262 / 0.13796 |

## 12.5 Results (Δ vs Arm A, 同 learner 同协议)

| Arm | 特征增量 | B51_60 Δ | B61_70 Δ | avg Δ | blendΔ (vs canon) | corr(A) | corr(canon) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | — | 0 | 0 | — | −0.000297 | 1.000 | 0.865 |
| B | +raw 73 | +0.005038 | +0.006639 | **+0.005838** | +0.000484 | 0.932 | 0.874 |
| C | +Z 73 | +0.005891 | +0.009139 | **+0.007515** | **+0.000941** | 0.924 | 0.872 |
| D | +raw+Z 146 | +0.006475 | +0.005393 | +0.005934 | +0.000618 | 0.919 | 0.871 |
| E | +raw+z² 130 | +0.004706 | +0.003900 | +0.004303 | +0.000347 | 0.927 | 0.869 |
| **D − E** | innovation − capacity | **+0.001770** | **+0.001493** | **+0.001632** | — | — | — |

**逐月 C-vs-A: 17/20 月正 (85%)**; 负月: m58 (−0.0003), m63 (−0.0132), m65 (−0.0104)。
**corr(C,B)=0.959**, corr(C,canon)=0.872, corr(B,canon)=0.874。

## 12.6 Monthly table (C vs A, 完整 20 月)

| month | A cos | C cos | Δ(C−A) | month | A cos | C cos | Δ(C−A) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 51 | 0.12744 | 0.13568 | +0.00824 | 61 | 0.10837 | 0.12172 | +0.01334 |
| 52 | 0.10306 | 0.11168 | +0.00862 | 62 | 0.12515 | 0.12667 | +0.00152 |
| 53 | 0.12744 | 0.13635 | +0.00890 | 63 | 0.13420 | 0.12098 | **−0.01322** |
| 54 | 0.12576 | 0.13394 | +0.00817 | 64 | 0.14011 | 0.14221 | +0.00210 |
| 55 | 0.11258 | 0.11953 | +0.00695 | 65 | 0.14692 | 0.13653 | **−0.01039** |
| 56 | 0.12463 | 0.13416 | +0.00954 | 66 | 0.16106 | 0.18600 | +0.02494 |
| 57 | 0.14881 | 0.14932 | +0.00051 | 67 | 0.12821 | 0.13718 | +0.00898 |
| 58 | 0.13370 | 0.13342 | −0.00029 | 68 | 0.13040 | 0.14575 | +0.01534 |
| 59 | 0.15478 | 0.15500 | +0.00021 | 69 | 0.11598 | 0.12255 | +0.00658 |
| 60 | 0.13627 | 0.14022 | +0.00396 | 70 | 0.10166 | 0.10895 | +0.00729 |

## 12.7 Ablation / 诊断

- **Nuisance quality** (§5.7): R²(O)=0.143, R²(T)=0.223 (均值); 34% 特征 R²>0.05 —
  状态确实预测事件流 (条件化有对象); `ob_iat_mean` R²=1.0000 / `tb_iat_mean` R²=0.992:
  与 mstate 的 o_avg_time_gap / t_avg_time_gap 同源 (重复特征, 其 Z≈0, 无害且符合预期)
- **capacity control**: D−E = +0.0016 双块为正 → innovation 增量**不是**更多特征容量所致
- **重要性**: Z 特征进 top-20 gain (Z_tb_sell_cnt #13, Z_tb_signed_vol, Z_ob_bid_cancel_cnt) —
  trade 成交侧的 surprise 是主要贡献
- **raw 家族独立发现**: Arm B (+0.0058) 说明 side×action 拆分等 73 个新 raw 聚合
  本身就有独立信息 (f0726 无 side 拆分) — 独立于 innovation 的新特征面

## 12.8 Failure analysis (负月 m63/m65)

两个月大幅为负 (−0.013/−0.010) 且都在 late 块 — 需诚实标注: late 整体 +0.0091 由 m66
(+0.0249) 与 m68 (+0.0153) 支撑, 去掉这两个月后 late 剩余约 +0.001。最可能类别:
**Signal exists but regime-uneven** (条件化增益在特定月份强, 两个月反转); 不能排除
**OOD failure** 的局部形态 (63/65 月事件流-状态关系漂移)。这不推翻主线 (双块 + 17/20 月),
但完整版必须检查月级稳定性门禁。

## 12.9 Decision

**SCFI CONTINUE** (任务书 §5.12):
- ΔC = +0.0075 ≥ +0.0005 ✓ (15× 门槛)
- Δ(D−E) = +0.0016 ≥ +0.0005 ✓ (3× 门槛)
- late (61-70) = +0.0091 > 0 ✓; 月度 17/20 = 85% ≥ 65% ✓
- 非单一特征/月份驱动 ✓ (top-20 中 3 个 Z 特征, 17 个月正)

**未达 LIVE** (corr(C,canon)=0.872 > 0.80), **未达 STRONG** (blend +0.00094 < +0.0015)
→ **§5.14 升级阶梯 (条件强度/点过程/FiLM) 仍被门禁**。

**关于 corr(C,B)=0.959 的说明** (§5.13 kill 条款的字面触发 vs 直接证据):
kill 条款 "innovation 与 raw 几乎完全等价" 的意图被直接证据反驳 —
C 比 B 高 +0.0017, D−E +0.0016, 且二者都在双块稳定; 高相关来自共享 152 特征主导的
公共 alpha, 不是等价。按 §9 纪律, 该条款标注为 caveat 而非自动 KILL,
判定取 CONTINUE (边界由直接 Δ 证据裁决)。

## 下一步 (受门禁约束)

1. **RealMLP spot-check** (learner 稳健性): 同一 152+Z 特征用生产 learner 重跑 A vs C —
   若增量存活, SCFI 结论 learner 无关 (LGB-only 是当前最大局限)
2. 通过后: blend 权重生产化 (canonical + C 臂, holdout 调权) — 可进入提交队列评估
3. §5.14 阶梯 (conditional count/scale → event intensity → point-process) 仍等 LIVE 门禁
