# MSCapital P5-A — Nested Temporal MAG-Gate Probe 报告 (2026-08-14)

> 实验: `scripts/p5a_mag_gate.py` | 产物: `output/p5a_mag_gate/{results.json, p5a_preds.npz}`
> 任务书: "P5-A Nested Temporal MAG-Gate Probe" (2026-08-14, §4)
> 判定: **KILL** (3/6 判据, 嵌套下无增益)

## 12.1 Hypothesis

Market 的幅度预测 m̂ (= E[|y||M], corr 0.43-0.47) 能否对冻结基线 p 做
**逐样本条件幅度分配** (pred' = a_bin(m̂)·p), 且该条件化在 NESTED temporal
协议下提升全局 cosine? —— **否, 实验证伪。**

## 12.2 Data

| 项 | 值 |
|---|---|
| p (基线) | canonical clean-baseline-v2 OOF (`canonical_residual_oof.npz`), months 21-70, 885,936 行, 每行 source_train_end < month (per-block refit, RealMLP+Table w=0.37) |
| m̂ (幅度 OOF) | MarketSeqNet \|y\| 头 — **P5-02I P_mag probe 完全同构复刻** (架构/seed 42/15ep/AdamW 1e-3/cosine loss), 仅 train 21-40 → 预测 41-70 |
| y | label.target (train 文件 months 0-70) |
| 窗口 | outer block 51-60 (gate fit 41-50) + 61-70 (gate fit 41-60); bin 阈值只在 gate-fit 窗口计算 |
| 非嵌套参照 | fit 41-50 → eval 51-70 (仅诊断, 不参与判定) |

m̂ 复现检查: corr(m̂,|y|) 41-50 = **+0.4664** (P5-02I: 0.466), 51-70 = **+0.4266**
(P5-02I: 0.427) — 幅度模型忠实复刻, 负结果不是模型损坏所致。

## 12.3 Leakage audit

- [x] gate (bin 权重 a_k) 只在 eval block 之前月份的 OOF 上拟合 (51-60 ← 41-50; 61-70 ← 41-60)
- [x] bin 分位数阈值只在 gate-fit 窗口内计算 (eval 样本的 m̂ 分布不参与)
- [x] m̂ 来自 21-40 训练的编码器, 41-70 全部未见 (严格 OOF)
- [x] p 来自 per-block refit 基线, source_train_end < 行 month
- [x] 模型选择 (a_k 优化) 只看 gate-fit 窗口的 cosine; eval 结果未参与任何选择
- [x] 非嵌套参照明确标注为诊断, 不进入判定

## 12.4 Baselines

| 基线 | 值 |
|---|---|
| canonical baseline OOF cos (51-70, 嵌套拼接) | 0.1338 附近 (逐月见 results.json monthly) |
| 非嵌套参照 (fit 41-50 → eval 51-70) | Δ = **+0.000138** (表面小幅正, 嵌套后消失) |
| 置换对照 (m̂ 在 gate 窗口内打乱) | Δ = **+0.000004** ≈ 0 |

## 12.5 Results (嵌套, 51-70 拼接)

| Model | Global cosine | Δ | Late (61-70) | Positive months | corr with base |
|---:|---:|---:|---:|---:|---:|
| canonical baseline | 0.13408 (nested concat) | — | — | — | — |
| + 4-bin gate (nested) | 0.13393 | **−0.000146** | +0.000005 | 6/20 | 0.9999 |
| 非嵌套参照 (诊断) | — | +0.000138 | — | — | — |
| 置换对照 (诊断) | — | +0.000004 | — | — | — |

## 12.6 Monthly table (完整 20 个月在 results.json; 首 5 月示例)

| month | n | cos_base | cos_new | Δ | norm_ratio |
|---:|---:|---:|---:|---:|---:|
| 51 | 17,852 | 0.14166 | 0.14155 | −0.000108 | 1.013 |
| 52 | 17,792 | 0.12209 | 0.12221 | +0.000121 | 1.007 |
| 53 | 17,768 | 0.14704 | 0.14684 | −0.000198 | 1.006 |
| 54 | 17,786 | 0.14462 | 0.14454 | −0.000082 | 1.006 |
| 55 | 17,771 | 0.13167 | 0.13154 | −0.000127 | 1.008 |

## 12.7 Ablation / sanity checks (§4.6)

| 检查 | 结果 | 判读 |
|---|---|---|
| A. gate 非常数? | a1=[0.992, 0.990, 0.992, 1.025], a2=[1.006, 0.998, 0.996, 1.000]; std=0.011 | **gate ≈ 常数** — 优化器找不到条件幅度形状 |
| B. gain 来源 | 月度 6/20 正; |y| 分位/ m̂ 分位分解见 results.json (无结构) | 无稳定来源 |
| C. extreme months | 去掉 top-1 gain month 后 Δ = −0.000151 (更负) | 无极端月驱动 |
| D. norm | norm_ratio_max = 1.01, 无爆炸 | 正常 |
| E. activity 分层 | lo −0.000117 / hi −0.000206 | 两段均负 |
| F. 置换对照 | +0.000004 ≈ 0 | 非嵌套的 +0.000138 也是通用 4 参数形状优化的产物, 非 magnitude 特异 |

## 12.8 Failure analysis

最可能类别: **Hypothesis false (条件幅度分配无法变现)**, 附带机制证据:

1. m̂ 信息真实 (corr 0.43-0.47 复现) 且 v7 方向精度随 |y| 单调 (D1-D3) — 两个前提都成立,
   但 **gate 优化器在嵌套窗口内找不到 a_k ≠ 1 的稳定解** (a≈1, gate_std=0.011):
   即"大波动样本方向更准"的增益, 在 cosine 全局归一化下**无法通过逐样本幅度加权兑现** —
   cosine 是全局尺度不变的, 只有跨样本相对形状变化才得分, 而 v7 的 |pred| 形状
   已经是当前可学的最优形状 (0.156 幅度相关虽弱, 但 gate 学到的形状比它更差)。
2. 非嵌套参照 +0.000138 → 嵌套 −0.000146: 经典 **Leakage removed optimistic gain** 模式
   (任务书 §3.2 预警的正是这个: 同一批样本拟合+评估 = 二级模型 in-sample)。
3. 不是 Representation failed: m̂ 质量被复现检查确认; 不是 OOD failure: 51-60 与 61-70
   行为一致 (都≈0)。

## 12.9 Decision

**KILL** (任务书 §4.7: Δ<+0.0004 且 gate 退化常数 且嵌套后增益消失)。
判定依据: Δ_outer=−0.000146 < +0.0004; 月度 6/20 < 70%; 去掉 top-1 月后更负;
gate std=0.0108 接近常数; 置换对照≈0。

**MAG-MoE (§4.8) 不启动** — 其前提 (gate 有信号) 被证伪。幅度调制完整版
(COC/log-|y| 头 + 尾部专家) 在同一逻辑下优先级下调: 若 4 参数条件化都无净增益,
更复杂的 MoE/双头架构的期望值更低 (除非换 target 训练口径 — 但任务书 §16
禁止为救理论改协议)。

## 遗留/备注

- 本实验只测了 **p 的幅度形状调制**; 未测"用 m̂ 重训幅度头" (那属于新目标建模,
  超出 P5-A 范围, 由 P5-B/C 结果后统一裁决)。
- seq_tmp.bin (7.6GB, 41-70 序列) 保留在 output/p5a_mag_gate/, P5-C 可复用其构建逻辑
  但窗口不同 (P5-C 用 last-10), 不共享。
