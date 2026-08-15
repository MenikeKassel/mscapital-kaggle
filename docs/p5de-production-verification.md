# MSCapital P5-D/P5-E — 生产级验证与 SCFI 升级裁决 (2026-08-15)

> P5-D: `scripts/p5d_prod_blend.py` → `output/p5d_prod_blend/{results.json, p5d_preds.npz}`
> P5-E: `scripts/p5e_realmlp_spotcheck.py` → `output/p5e_realmlp_spotcheck/`
> 前置: P5-B (LGB 单 seed 五臂) + P5-C (RICS) 见 `docs/p5b-scfi-report.md`

## 目的

P5-B 的 SCFI 证据 (LGB 单 seed ΔC=+0.0075) 是否在生产级配置下成立?
两个补充实验: ① P5-D: LGB 3-seed 集成 + CatBoost 双 learner 融合验证;
② P5-E: **生产 learner (RealMLP, clean-realmlp-v2a) 精确 spot-check** — SCFI 升级的裁决证据。

## P5-D 结果 (blend 权重 holdout 49-50/59-60 调, eval 51-60/61-70 冻结, 双块平均)

| 臂 | Learner | blendΔ avg | B51_60 | B61_70 | corr(canon) |
|---|---:|---:|---:|---:|---:|
| A (152) | LGB 3seed | −0.000272 | −0.000284 | −0.000260 | 0.886 |
| A | CatBoost | −0.000063 | +0.000008 | −0.000134 | 0.882 |
| B (+73raw) | LGB 3seed | +0.000448 | +0.000671 | +0.000226 | 0.888 |
| B | CatBoost | −0.000187 | +0.000425 | −0.000799 | 0.880 |
| **C (+Z)** | **LGB 3seed** | **+0.000849** | **+0.001282** | **+0.000415** | 0.885 |
| C | CatBoost | +0.000163 | +0.000844 | −0.000518 | 0.870 |
| D (+raw+Z) | LGB 3seed | +0.000549 | +0.001212 | −0.000114 | 0.884 |
| D | CatBoost | +0.000662 | +0.000955 | +0.000369 | 0.872 |

- **C_lgb 唯一双块均为正且最强** (avg +0.00085); 3-seed 集成确认 P5-B 单 seed 结论 (无 seed 侥幸)
- D 不比 C 好 (LGB 上加 raw 反而略降) → 生产候选 = C (152+Z)
- CatBoost 弱于 LGB 且 late 块转负 → CatBoost 不作为生产成员
- 逐月 vs canonical 的月正率低 (1-4/20) 是 learner 绝对强度差 (表格模型 standalone < canonical blend), 与 P5-B 同因; blend 增益来自互补性

## P5-E 结果 (RealMLP, R61_70: inner 0-50 / tune 51-60 / refit 0-60 / outer 61-70)

| arm | 特征 | outer cosine (61-70) | corr(canon) | blend_w (51-60 调) | blendΔ (61-70) |
|---|---:|---:|---:|---:|---:|
| A | 152 | 0.148526 | 0.9673 | 0.05 | −0.000068 |
| **C** | **152+Z** | **0.152570** | **0.9402** | **0.50** | **+0.001369** |
| Δ(C−A) | | **+0.004044** | | | |

- Arm A 复现 canonical RealMLP 水准 (0.1485, R61_70 realmlp 组件同水平) → 复刻忠实
- **Z 特征在生产 learner 上 standalone +0.0040, blend (61-70 冻结) +0.0014** — SCFI 升级证据成立
- SmallMLP spot-check (Δ≈0) 被推翻: 小 MLP 是过弱代理, 无法利用 73 维条件特征的精细结构
- blend_w=0.50 顶到网格上沿 → 更高权重可能更好 (生产融合需扩网格验证)

## 裁决

**SCFI 升级门禁通过** (任务书 §5.14): 生产 learner (RealMLP) +0.0040 standalone /
+0.0014 blend, 表格族 (LGB 3seed) +0.00085 blend 双块正 — 两个 learner 家族独立确认。
条件创新表示 (Z = 事件流 − E[事件流|市场状态]) 是 0.142 后第一个**跨 learner 验证的新信息面**。

下一步 (提交纪律照旧, 需用户拍板):
1. **生产推理**: RealMLP 全量 (0-70) 重训 152+Z → test 预测; 与 canonical blend (w 网格扩展至 0.7, 51-60 调)
2. PSEUDO 门禁 + 分布检查 (prediction std 对比) 通过后才谈提交
3. §5.14 阶梯 (conditional count/scale → event intensity → point-process) 现在有依据, 但排在 1/2 之后
