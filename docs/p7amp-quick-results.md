# P7-AMP 快速版结果 (2026-08-15)

> 协议: GPT1 round2 的 P7-AMP 快速版 (arm A/C/D, 嵌套: gate+α 拟合 m21-50, frozen 评估 m51-70)
> 脚本: `scripts/p7amp_quick.py` → `output/p7amp_quick.json`
> 前置: `scripts/p7amp_audit.py` (canonical OOF 885,936 行 × 5 amplitude 特征)

## 结果

| 臂 | 内容 | α* | cal cosine | frozen cosine | Δ vs A |
|---|---|---|---|---|---|
| A | baseline 原样 | — | 0.14203 | **0.14849** | — |
| C | gate(不含 mid_range): mid_std/n_order/n_tx/depth1 | 0.0 | 0.14203 | 0.14849 | +0.00000 |
| D | gate(含 mid_range + 上述) | 0.0 | 0.14203 | 0.14849 | +0.00000 |

*α 在 calibration 选一次 (嵌套); 全部选 0.0 = gate 无增益

**主结果**: ΔD(vs A) = **+0.00000** → **RED 判定** (< +0.0005)

## α 曲线 (确认非局部极值, 单调下降)

| α | cal | frozen |
|---|---|---|
| 0.00 | 0.14203 | 0.14849 |
| 0.10 | 0.14092 | 0.14802 |
| 0.25 | 0.13711 | 0.14531 |
| 0.50 | 0.12815 | 0.13835 |
| 1.00 | 0.10841 | 0.12292 |

## 失败机制 (gate 分桶诊断)

| gate 权重分位 | n | cos(p,y) | \|y\|mean |
|---|---|---|---|
| 低 50% | 176,623 | **+0.215** | 0.0013 |
| 50-80% | 105,973 | +0.110 | 0.0019 |
| 80-90% | 35,324 | +0.109 | 0.0025 |
| 90-95% | 17,663 | +0.116 | 0.0030 |

**机制**: gate 权重高 (=预测波动大) 的样本, 方向预测质量反而**更差** (cos 0.215 → 0.11)。
放大高波动样本 = 放大方向噪声 → cosine 下降。baseline 的幅度分布**已经隐式最优** (不给大波动样本加权)。

## 附加发现

- corr(mid_range, m_mid_std) = **0.196** — mid_range 携带 f0726 已有波动特征之外的信息, 但该信息对 cosine 无用 (幅度侧已被 gate 路线证伪)
- corr(mid_range, m_rv) = 0.076
- gate 本身能预测幅度 (corr(g,|y|)=0.383) — **幅度可预测 ≠ 幅度加权有效**

## 判定

**RED** — 关闭:
- [x] market amplitude statistics (34 特征 已有零信号, 双重确认)
- [x] mid_range / mid_std variants
- [x] volatility confidence calibration (gate 路线)
- [x] "Market → Confidence" 假设 (GPT1 round2 核心主张, 实测证伪)

**结论 (与 GPT1 预注册的 RED 路径一致)**: baseline 已隐式吸收 heteroskedasticity;
"方向×幅度分解" 在 cosine 下不成立 — 因为方向质量随波动下降, 幅度加权必然有害。
资源转向: **Order→Transaction lag response (GPT P1.5)** — 与 P5-02I "跨通道同步是核心" 结论同源, 是当前唯一有独立机制理由的方向。

## 遗留 (未跑)

- arm B (152 + mid_range → RealMLP 直接加特征): corr 0.2 显示有独立信息, 但 gate 机制已证伪幅度价值, RealMLP 重训 ~1-2h, GPT 自身预期低 → 不建议单独跑
