# P6R Repository Audit — Retrieval-Conditioned Residual Alpha

> 日期: 2026-08-14 | 执行: Hermes (P6R 首席研究员)
> 方法: 逐文件核对实际代码 + 当前 artifact；SOURCE OF TRUTH 优先级 = 实际代码/artifact > README > 实验结果文档 > plan > 历史文档。
> 本审计是 P6R 系列的前置；任何冲突以实际代码与 artifact 为准并在此明示。

## 0. 仓库定位与版本

- 本地仓库: `D:\mscapital-kaggle`（`E:\aiworkspace\mscapital-kaggle` 为空目录，已被 D 盘取代）
- 当前分支: main（HEAD b61a852）；提交冻结纪律：代码/文档入库，数据/权重/预测不入库
- 数据: `D:\mscapital-forecasting\data\`（raw/ 四文件 + processed/ 特征 parquet）
- 路线文档: `docs/plan-v1.8.0.md` 为唯一 current source of truth（v1.0–v1.7 已归档 `docs/_archive/plans/`）；**`docs/plan-v1.4.0.md` 已不存在**（任务书所列旧路径），其内容被 v1.8.0 吸收

## A. 当前真正的 frozen baseline

**Clean Baseline v2（协议-v2 冻结的生产基线）**，两个层次：

1. **canonical rolling OOF**（残差研究唯一允许的 OOF 基线）
   - Artifact: `output/canonical_residual_oof/canonical_residual_oof.npz`（+ `manifest.json`，config_hash 可重放校验）
   - 覆盖: **month 21–70，885,936 行**，5 个 expanding block，每行满足 `source_train_end < month`（严格 OOF，非重叠平均）
   - Block 定义（splits.py `CANONICAL_ROLLING_SPLITS`）: R21_30 (refit 0–20 → 21–30), R31_40 (0–30 → 31–40), R41_50 (0–40 → 41–50), R51_60 (0–50 → 51–60), R61_70 (0–60 → 61–70)；每 block 的 RealMLP/Table 独立训练，block 内 RMS scale 从该 block 的 inner-tune (11–20/21–30/31–40/41–50/51–60) 拟合
   - 融合规则（`apply_production_rule`）: `0.63·RealMLP/rms_realmlp + 0.37·Table/rms_table`（frozen: method=rms, table_weight=0.37, PRODUCTION_TABLE_WEIGHT）
2. **frozen production baseline（逐 outer 冻结）**
   - 路径: `output/c4_protocol_closed_final/clean-baseline-v2/{PSEUDO,H2,T3,T4}/`（manifest status=frozen，predictions.npz split=`:production_default`）
   - PSEUDO frozen: train 0–32 / valid **33–70**（672,948 行），production_schema_replay_cosine = **0.1425503399680412**（= E01 表内 "Frozen baseline PSEUDO"）
   - H2: 0–40 / 51–60；T3: 0–50 / 51–60；T4: 0–50 / 61–70
   - 生产 scale: `output/c4_frozen_final/clean-baseline-v2/production/production_scales.json`（scale_realmlp=0.01481242584, scale_table=0.0003505723161，fit 于 canonical OOF m51–70）

**⚠️ 审计发现（重要）**：canonical OOF 基线 ≠ c4-frozen 生产基线（同月份余弦不同）：

| outer (月份) | canonical OOF cos | c4-frozen cos | 差 |
|---|---|---|---|
| PSEUDO (33–70) | **0.145825** | 0.142550 | +0.003275 |
| H2 (51–60) | 0.143515 | 0.141862 | +0.001653 |
| T3 (51–60) | 0.143515 | 0.143549 | −0.000034 |
| T4 (61–70) | 0.154759 | 0.157053 | −0.002294 |

原因: canonical OOF 逐 block 用更晚的 refit（如 month 61–70 由 refit 0–60 预测），而 c4-frozen 用单次 refit（PSEUDO 为 0–32）。E01/E02 的惯例 = 残差目标来自 canonical OOF，最终 blend 锚在 c4-frozen 生产基线上。**P6R 沿用此接缝**（主评估锚 c4-frozen 以便与 E01/E02 直接可比，同时报告 canonical 锚定诊断）。

## B. v7 OOF prediction 的准确路径、覆盖月份、sample_id 对齐

- "v7 OOF" 有两个含义，均已核实：
  1. **v7 PSEUDO（单折）**: `output/rlps_v12/realmlp_pseudo_pred.npz`（keys: pred/y）+ `output/rlps_v12/v5_table_pseudo_pred.npz`；v7 = 0.8·table + 0.2·RealMLP，覆盖 **m33–70, 672,948 行**（与 canonical PSEUDO frozen 月份一致）；sample_id 对齐 = 与 label.feather 的 month 33–70 子集按 sample_id 排序一致
  2. **v7 式 rolling OOF（残差研究用）**: 项目不保留逐月 v7 滚动 OOF artifact；残差研究统一使用 **canonical_residual_oof**（Clean Baseline v2 rolling OOF，21–70）。P5-01 用 `0.8·v5_c + 0.2·rl_c` 构造 "v7_like" 作近似（33–70 对齐 canonical 行）
- 对齐机制: 所有 artifact 均以 `sample_id` 全局唯一为契约（`CanonicalOOF.validate` 强制），特征取行用 `argsort(sample_id) + searchsorted`（`_take_features`/`_take_context`）

## C. canonical rolling OOF 的实现方式

- 源码: `src/mscapital/residual.py`（build_clean_baseline_oof_block → build_canonical_oof → write_canonical_oof_artifact → load_canonical_oof_artifact）
- 校验链: block manifest（experiment_id/config_hash/data_fingerprints/行数/月份/逐数组 hash）→ 合并后全局 sample_id 唯一、`source_train_end < month` 全量 assert、month 21–70 全覆盖、每月的 source_train_end 与注册表精确一致
- 每 block 的 RMS scale 只从该 block 的 inner-tune 预测拟合（不重放 m51–70 生产 scale 到早期 block）
- **frozen outer views**（`visible_oof_end`）: PSEUDO → month ≤ 32；H2 → ≤ 40；T3/T4 → ≤ 50（T4 禁读 51–60）。残差模型可见训练区 = view；冻结评估区 = NESTED_SPLITS outer_valid
- E01/E02 的 inner split（`RESIDUAL_INNER_SPLITS`）: PSEUDO train 21–26 / tune 27–32；H2 21–30 / 31–40；T3/T4 21–40 / 41–50

## D. E02 context feature 的准确字段和 artifact

- E02 的 11 个 context 特征（`features/revol_lite.py: context_feature_names()`）:
  - 6 CONTEXT: `revol_log_ret_sigma_60, revol_log_depth_rms_60, revol_log_order_volume_rms_60, revol_log_trade_volume_rms_60, revol_log1p_order_event_rate_60, revol_log1p_trade_event_rate_60`
  - 5 CONTEXT_STATE: `revol_mid_net_z_60, revol_spread_mean_z_60, revol_depth_change_z_60, revol_order_flow_z_60, revol_trade_flow_z_60`
- Artifact: `output/e01_revol_lite_features/revol_lite_train.parquet`（manifest experiment_id=e01-revol-lite-features, status=complete；37 特征 = 7×4 窗口 + 11 context + 3 missing；1,257,637 行 = 全部 train month 0–70；**100% 覆盖 canonical OOF sample_id** ✓ 已实测）
- 构建: 纯样本内 look-back 窗口（5/15/30/60s），无 label 拟合、无跨样本统计量（leakage-safe，源码注释明示）
- E02 结果（`output/e02_context_shift/context_shift.json`）: HistGB pooled residual cosine **0.013223925**，bootstrap CI [+0.00836, +0.01852]，4/4 折正，worst +0.00972；Ridge 仅作线性基准。**gate 通过（只开放 E05 learned retrieval 注册，不构成提交候选）**
- E02 残差定义: `r = y − β·baseline`，β = dot(p,y)/dot(p,p) 每折在 train 月拟合（⚠️ β≈0.0004，因生产基线 RMS≈0.75 而 target std≈0.0026 —— 残差 ≈ y 的标度化，这是既定惯例，P6R 沿用并报告 β）

## E. P5-01 market encoder / latent 是否已有可复用 artifact

**没有可直接复用的 latent/encoder artifact。**
- `output/p5_01_market_sequence/` 只有 `results.json` + `seq_tmp.bin`（memmap 临时序列矩阵）；`output/p5_02i_info_audit/` 同样只有 results.json + seq_tmp.bin
- `scripts/p5_01_market_sequence.py`: 模型（Conv1D 64ch k=7 ×2 residual blocks → GAP → 32d → 1）仅内存中使用，**未保存权重**；seq_tmp.bin 是构建中间态，无 manifest、无 hash
- 已知结果（results.json）: MSE 臂 corr(y)=−0.0013 / Δfrozen=−0.00024；cosine 臂 corr(y)=+0.086, corr(v7_like)=0.492, corr(resid)=−0.359, α=0.17（41–50 选）, frozen 51–70 Δ=+0.000926 (17/20 月正)，lo 活动 +0.0021 vs hi +0.0006
- 输入口径: 200 步 × **18 通道**（11 raw + 7 derived: mid/spread1/depth1/imb1/spread2/depth2/imb2），均匀 3s 网格 carry-forward
- **P6R-04 如需 market latent 必须重训**（从 market.feather 重建序列，训练/冻结协议同 P5-01：21–40 训练、41–50 选 α、51–70 冻结），并补存 frozen encoder

## F. 152 features 的准确来源和 artifact

- Artifact: `D:\mscapital-forecasting\data\processed\f0726_train.parquet`（154 列 = sample_id + 152 特征 + target；1,257,637 行 = 全 train）；test: `f0726_test_f32.parquet`
- **100% 覆盖 canonical OOF sample_id** ✓ 已实测
- 特征命名族: t_*（成交动力学）、o_*（订单）、m_*（market）、x_*（cross/事件序列），构造来自公开 RealMLP 方案的 0726 特征复刻（脚本 `scripts/40_build_0726.py` 体系）
- 模型: RealMLP_RQ (n_ens=16) 单模型 PSEUDO 0.138560；v7 融合 0.139683 → LB 0.135

## G. 数据可直接复用 vs 必须重新生成

| 资产 | 路径 | 状态 |
|---|---|---|
| canonical residual OOF (21–70) | output/canonical_residual_oof/* | ✅ 直接复用（manifest 校验后） |
| c4 frozen 生产基线 4 outer | output/c4_protocol_closed_final/clean-baseline-v2/{PSEUDO,H2,T3,T4}/ | ✅ 直接复用 |
| E02 11 context 特征（37 全量） | output/e01_revol_lite_features/revol_lite_train.parquet | ✅ 直接复用 |
| 152 特征 | data/processed/f0726_train.parquet (+test) | ✅ 直接复用 |
| 原始四文件 | data/raw/{label,market,order,transaction}.feather | ✅ 存在（market 600s 序列重建所需） |
| P5-01 market latent / encoder | — | ❌ 必须重新训练（未保存权重） |
| market 600s 序列矩阵 | output/p5_01_market_sequence/seq_tmp.bin | ⚠️ 仅临时件，非 frozen artifact；重训时重建 |
| test 预测/submission | output/submissions/* | 本轮禁止使用 |

## H. 文档与代码的不一致（明确列出）

1. **plan-v1.4.0.md 不存在**（任务书路径）→ 已归档，current = `docs/plan-v1.8.0.md`；其队列（P5-02M → B-lite v2 → P6-04 hard experts）状态 = "讨论定稿，未获执行许可"；**P6R 是用户新授权的执行主线**，与 plan 的 P2 (Residual Market-State KNN) / P6-04 (hard regime experts) / P6-01 (residual target, 已降 B) 重叠——P6R-00 ≈ P2 的严格化版本（状态表示改用 E02 已验证的 11 context 特征）
2. **market 通道数**: 历史文档 "13ch" 是 P5-01 之前的旧规划残留；README 已声明并统一为 **18 通道**；`p5_01_market_sequence.py` 实际 18ch ✓ 与 plan-v1.8.0 一致（旧 13ch 仅存于 _archive）
3. **"PSEUDO" 双义**: 传统 PSEUDO = train 0–32 / valid 33–70（OUTER_SPLITS，672,948 行）；协议-v2 残差研究的 "PSEUDO visible view" = months 21–32（visible_oof_end）。E03 的 "PSEUDO month 33–70" = 冻结评估月份。P6R 明确区分 selection（21–32）与 frozen eval（33–70）
4. **canonical OOF 基线 vs c4-frozen 生产基线分数不同**（见 A，±0.003 量级）——E01/E02 文档未明示此接缝；P6R 双锚报告
5. **c4 系列目录重复**（c4_formal / c4_formal_final / c4_protocol_closed / c4_protocol_closed_final / c4_frozen_final 均存在）——E01 实际用的 frozen baseline root = **c4_protocol_closed_final**（status=frozen、production_schema_replay_cosine 与 E01 数字吻合）；c4_formal_final 是 nested-complete 旧态
6. **β≈0.0004 的残差定义含义**（E02/outer_residual 惯例）未被文档解释：生产基线 RMS≈0.75（归一化后）与 target std≈0.0026 的尺度差导致 r≈y−0.0004·y0；不影响方向性结论但解释 residual cosine 时要引用原始尺度
7. README 声称 v7 valid/test std 比 0.7447 等校准数字与 RESULTS.md/calibration.md 一致 ✓；未发现第三处矛盾

## 特别检查结论

- ✅ plan market 通道数（18）与 P5-01 实现一致（13ch 为归档残留）
- ✅ 未发现任何 latent/特征使用 future month 信息（ReVol-lite 纯 look-back；canonical OOF source_train_end<month 强制）
- ✅ OOF predictions 严格 out-of-fold（逐 block refit 先于预测月，manifest 全链路 hash 校验）
- ✅ 特征 preprocessing 无跨 validation/test 拟合（ReVol-lite 样本内窗口；152 特征构建为事件级；P5-01 的 per-channel 标准化只从 21–40 拟合）
- ⚠️ 当前 "residual" 定义 = `y − β·y0`（β 每 outer 从可见 view 拟合，outer_residual()），**不是** y − y0；P6R 沿用为 primary 并报告 β=1 变体
- ⚠️ PSEUDO selection 区最早月份（21–22）在 canonical OOF 内邻居极少/为空（OOF 起点=21）——检索支持从 month 22 起才有 ≥1 个月邻居；α 选择（27–32）不受影响

## P6R 由此确定的数据契约

1. 残差目标: canonical OOF rows，r = y − β·y0（β 每 outer 从可见 view 拟合；β=1 作变体）
2. 邻居库: canonical OOF rows，**month < query_month 严格断言**（V1 主设计=全历史扩展；V2 变体=上限 visible_oof_end）
3. 状态表示: E02 11 context 特征（revol_lite_train.parquet）
4. α 选择: RESIDUAL_INNER_SPLITS tune 月（PSEUDO 27–32 / H2 31–40 / T3,T4 41–50），grid {0.05,0.10,0.15,0.20,0.30}
5. 冻结评估: NESTED_SPLITS outer_valid（PSEUDO 33–70 / H2 51–60 / T3 51–60 / T4 61–70），主锚 c4-frozen 生产基线（与 E01/E02 可比），辅锚 canonical OOF 基线
6. 融合公式（E01 惯例）: y_hat = RMS(y0) + α·RMS(r_hat)
