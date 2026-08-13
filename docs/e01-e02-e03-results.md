# 首批新方法实验结果（2026-08-13）

本记录对应 E01 ReVol-lite、E02 Reconditionor-lite 和 E03 稳定性审计。所有结果都来自 `output/`，没有读取 test target，也没有生成 Kaggle submission。

## E01 ReVol-lite

特征 artifact：`output/e01_revol_lite_features/revol_lite_train.parquet`

- 1,257,637 个样本，37 个固定宽度 `float32` 特征。
- 四折均使用预注册候选 `37-feature normalized+scale`，Residual CatBoost、inner early stopping、RMS normalization 和 alpha grid 固定。

| Outer | Frozen baseline | ReVol-lite | Delta |
|---|---:|---:|---:|
| PSEUDO | 0.142550340 | 0.143646991 | +0.001096651 |
| H2 | 0.141861992 | 0.144178045 | +0.002316052 |
| T3 | 0.143549308 | 0.144934229 | +0.001384921 |
| T4 | 0.157053101 | 0.159012794 | +0.001959693 |

Protocol-v2 基础 gate **未通过**：PSEUDO delta 为 `+0.001096651`，低于预注册的 `+0.0015`。因此 E01 不晋级为提交候选，也不进入融合。

## E03 稳定性审计

主审计只使用 PSEUDO 的 month 33–70，并以整月充分统计量做 5000 次 bootstrap（seed `2026`）。

- positive-month ratio：`0.842`
- mean monthly delta：`+0.001196252`
- worst monthly delta：`-0.002448567`
- delta slope：`+6.23e-06 / month`
- top-3 positive-month concentration：`0.265`
- aggregate delta 95% CI：`[+0.000599579, +0.001669554]`

稳定性附加 gate 通过，但合并 E01 基础 gate 后最终决策仍为 **失败**，原因仍是 PSEUDO 基础门槛。

## E02 Reconditionor-lite

E02 使用 11 个 E01 上下文特征，严格前向四折；month 未作为模型输入。Ridge 只作线性解释基准，HistGradientBoosting 为固定非线性诊断器。

- HistGB pooled residual cosine：`0.013223925`
- HistGB bootstrap 95% CI：`[+0.008356664, +0.018521100]`
- positive folds：`4/4`
- worst fold cosine：`+0.009721095`
- pooled normalized MSE improvement：`-0.000599439`

E02 gate **通过**。这只表示未来可以注册 E05 learned retrieval；E02 本身不是提交候选，也不与 Clean Baseline 融合。

## 产物

- E01 features：`output/e01_revol_lite_features/`
- E01 fold predictions：`output/e01_revol_lite/`
- E01 summary：`output/e01_revol_lite_summary.json` 和 `.md`
- E03 audit：`output/e03_stability/`
- E02 diagnostic：`output/e02_context_shift/`

