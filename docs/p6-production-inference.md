# MSCapital P6 — 生产推理：RealMLP-C test 预测 + PSEUDO 门禁 + blend 候选 (2026-08-15)

> 脚本: `scripts/p6_build_test_features.py` / `p6_prod_realmlp.py` / `p6_finish_test.py`
> 产物: `output/p6_prod/{realmlpC_test_pred.npz, realmlpC_pseudo_pred.npz,
> blend_candidates.npz, submission_candidate_p6.csv, results.json}`
> 前置: P5-D/E (SCFI 生产级确认, `docs/p5de-production-verification.md`)

## 管线

1. **test raw O/T 聚合** (73 特征, 复用 P5-B 的 chunk 函数, 25s)
2. **Z_test**: nuisance (M→O, M+O→T, Ridge, MAD 尺度) fit on train 0-70 → test。
   M = f0726 子集 (已验证与 m05 market_state 值级 100% 一致); f0726_test 的
   ~2% NaN 用 train 列中位数插补
3. **RealMLP-C test 预测**: preprocessor fit on train 0-70 (152+73Z) →
   refit (progress=1.0, 1334s) → test (647,896 行)
4. **PSEUDO 门禁**: 同构复刻 canonical PSEUDO 切分 (inner 0-20/tune 21-32/
   refit 0-32/eval 33-70)
5. **生产 blend**: v8b 锚 + RealMLP-C test, w 网格 0.30-0.70 (+0.55 主候选)

## 结果

### 分布检查 (v6 教训门禁)
- std_test / std_valid(61-70) = **0.9726** — 无尺度异常 ✓

### 生产 blend 权重 (51-60 调, 网格扩到 0.75)
- **w = 0.55** (score 0.144596) — 印证 P5-E 的 0.50 顶格信号, 扩网格后确认更优

### PSEUDO 门禁 (eval 33-70, 672,948 行)
| 指标 | 值 |
|---|---|
| RealMLP-C standalone | 0.139248 |
| canonical PSEUDO | 0.142550 |
| standalone Δ | −0.003302 (单模型 < canonical blend, 预期) |
| **blendΔ (w=0.75 调于 21-32)** | **+0.001435** |
| **跨验证窗口一致性** | R61_70: +0.001369 / PSEUDO: +0.001435 — 两个独立窗口 ≈ 同值 |

### 提交候选 (未提交)
- `output/p6_prod/submission_candidate_p6.csv` = rms(v8b) + 0.55·rms(RealMLP-C)
- 候选 w 网格: 0.30/0.40/0.50/0.60/0.70/0.55 全落盘 `blend_candidates.npz`
- **遵守提交纪律: 未自动提交 Kaggle, 等用户拍板**

## 判读

1. **RealMLP-C 的 blend 增益跨窗口稳定** (+0.0014, R61_70 与 PSEUDO 一致) —
   这是 0.142 后第一个带双窗口验证的新增量; LB 期望 ≈ 0.142 + 0.001×α
   (α 取决于 lb142 外部成分与 test 的相关结构, 无法本地精确估计)
2. **两个调权窗口都顶到/接近网格上沿** (PSEUDO w=0.75, 51-60 w=0.55) —
   若提交, 值得再扩展 w 到 0.8-1.0 做敏感性 (本期已在网格内给出候选)
3. PSEUDO standalone < canonical 是正常的 (单模型 vs 生产 blend),
   判定只依赖 blendΔ — 互补性才是 RealMLP-C 的价值
4. 未做 Kaggle 提交; 若用户决定提交, 候选 CSV 已就绪
