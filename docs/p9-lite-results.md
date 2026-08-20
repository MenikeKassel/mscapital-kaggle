# P9-Lite 三探针 Gate 结果 (2026-08-20)

> 用户拍板: 跑三个最小 Gate — **Cancel Pressure / Event-Time / M55-lite**, 不做 IC 大筛查。
> 目标: 回答 **152 baseline 是否丢失「事件类型 / 事件节奏 / 流动性撤退」三类微观结构信息**。
> 协议: C-05 E0 同款 (m0-32 train / m33-70 eval, fold-local robust+clip, 3×256 MLP, MSE, 30ep,
> seed 2026, best-cosine-epoch 选于 eval 33-70), **只改变 feature information**。
> 审判窗: **frozen 51-70** (uncentered cosine)。

## 基线复现

- P9-A/B/C 各 base 臂 frozen 51-70 = **0.1143636**, eval 33-70 = **0.113261** — 与 C-05 E0 / P9-NEUT γ=0
  参考值 (0.114364 / 0.113261) 逐位一致 → 协议无偏差。

## 结果总表

| 探针 | 新特征 | Δ eval 33-70 | **Δ frozen 51-70** | pos 51-70 | Δ月正 | pred corr | 判定 |
|---|---|---|---|---|---|---|---|
| **A Cancel** | 13 (side-split 撤单量/数、cancel pressure、近端撤单、触价撤单、relative to depth) | **+0.004378** | **+0.004146** | 20/20 (聚合) | 13/20 | 0.76 | 🟡→**强信号但 regime 集中** |
| **B Event-Time** | 18 (iat cv/log、burst_ratio、recent/prev rate、大单尾部) | −0.002754 | **−0.003887** | 20/20 | 4/20 | 0.84 | 🔴 **RED** |
| **C M55-lite** | 10 (DWI L1/L2、imb2 结构、trade entropy、DWI×entropy) | +0.000704 | **+0.000542** | 20/20 | 12/20 | 0.87 | 🟡 边缘 |

## P9-A Cancel Pressure — 真实有效, 但 regime 集中

- **+0.0041 frozen / +0.0044 eval** — 量级远超 GREEN 门禁 (+0.001), 且与 Z 绿灯同向:
  152 基线**无按 side 拆分的撤单量/压力** (只有 o_sec_cancel_new_ratio/volume 总量), 本探针补上的
  side-split 撤单不对称 (cancel_press_vol/cnt) 是真正的新信息。
- **月度**: 聚合 20/20 正, 月度 Δ 13/20 正 (top: 52 +0.026 / 67 +0.022 / 62 +0.020; 负: 59 −0.017 / 63 −0.015 / 66 −0.014)。
- **⚠️ activity regime 分裂 (关键)**:

  | regime | base | feat | Δ |
  |---|---|---|---|
  | low_act | 0.12372 | 0.11735 | **−0.0064** |
  | hi_act  | 0.11012 | 0.12043 | **+0.0103** |

  增益**几乎全部来自高活动样本**, 低活动样本被撤单特征拖累 −0.006。低活动样本撤单并不稀疏
  (中位撤单量 13,700, 压力存在 98.7%), 因此**不是数据缺失伪影**, 是真实的"流动性撤退在高活动市
  场才有信息"的条件效应 — 与 brief 提出的"persistent microstructure alpha 在活动节奏里"一致。
- **判定**: 按用户三条件 (Δ≥+0.001 ✓ / 多数月正 ~65% ▲ / **regime 稳定 ✗**) → **非 clean GREEN**。
  定性为 **YELLOW-强信号**: 量级远超大, 但 regime 集中, 需要像 NEUT 一样先进联合/校准验证
  (或特征层做 cancel-activity 条件化) 再谈叠加。与 Z (152+73Z GREEN +0.0060, 含同源 `Z_ob_cancel_side_imb`)
  互为归因证据: **Z 绿灯的一部分可能正来自 cancel side-imb 这一族**。

## P9-B Event-Time — RED (基线已覆盖)

- **−0.0039 frozen / −0.0028 eval**, 月度 Δ 仅 4/20 正 (70 月灾难 −0.044), 叠加明显有害。
- **失败机制 (与用户预测的 RED 原因逐字一致)**: 152 基线**已含事件节奏信息** — `o_avg_time_gap /
  o_time_gap_std / o_sec_rowcount_near_far_ratio_15/30/60 / t_sec_rowcount_near_far_ratio` 等近/远窗口
  行数比就是 burstiness/acceleration; iat cv/log 与**未条件化的原始聚合**只是冗余噪声变体, 加倍反而伤。
- 注意与 Z 的对照: Z 中**条件化 (O−E[O|M]) 后的** iat/burst 是 GREEN; 这里测的是**原始聚合**, 为负。
  → "事件节奏信息" 确实存在 (Z 证明), 但它必须经 market/tx 条件化, 原始形式不提供增量的结论成立。

## P9-C M55-lite — YELLOW 边缘

- **+0.0005 frozen / +0.0007 eval**, 12/20 月 Δ 正, 方向稳定; regime 略偏 hi_act (+0.0008 / −0.0004)。
- DWOFI 只有 L1/L2 两层可构造 (原式 Depth-5 不可复刻), 熵项与 t_buy_ratio 高度重叠 → 边缘符合预期,
  与 P10-FM M1 L2 档位 (+0.0008 边缘) 一致。**YELLOW → 进联合实验**, 不作为独立候选。

## 结论 (回答 brief 的核心问题)

> **152 baseline 丢失了什么?** — 明确丢失的是「**按 side 拆分的流动性撤退 (撤单) 不对称**」
> (P9-A +0.0041, 但仅高活动), 没有丢失「事件节奏」 (P9-B RED, 基线已覆盖)。DWI/熵为边缘。
> 同时给出归因推论: Z (152+73Z) 绿灯中很可能有相当份额来自 cancel side-ib 一族
> (P9-A 与其同源、单测即 +0.0041)。

## 复现

```
.venv/Scripts/python.exe scripts/p9_lite_build.py --pkg all      # 特征构建 (~3min)
.venv/Scripts/python.exe scripts/p9_lite_train.py --pkg A --arm base|feat   # 训练 (~65s/跑)
.venv/Scripts/python.exe scripts/p9_lite_report.py                          # 汇总
产物: output/p9_lite/pkg{ABC}/{base,feat}/{results.json,preds.npz,best_cos.pt}
```
