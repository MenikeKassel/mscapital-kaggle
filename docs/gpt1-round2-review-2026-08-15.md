# GPT1 第二轮评审裁决 (2026-08-15)

> 输入: GPT1 基于全量实测的 EDA 修正 + P7-AMP 完整 protocol
> 方法: 逐条实测验证 (canonical OOF 885,936 行 × 5 amplitude 特征审计), 对照本地台账
> 裁决符号: ✅ 实测支持 / ⚠️ 部分 / ❌ 修正 / 🔍 新发现
> 审计脚本: `scripts/p7amp_audit.py` → `output/p7amp_audit.parquet`

## 一、核心论断验证 (P7-AMP Step 2 实测审计)

**机制问题**: baseline 是否已隐式吸收 heteroskedasticity?

| 审计量 | 实测值 | 解读 |
|---|---|---|
| corr(\|p\|, \|y\|) | **+0.068** | baseline 幅度校准几乎为零 (1.0=完美) |
| corr(mid_range, \|y\|) | **+0.371** | 复现 EDA +0.377 ✅ |
| corr(mid_range, \|p\|) | +0.105 | 已反映但远低于应有 0.37 |
| corr(mid_std, \|y\|) | +0.338 | 同上 |
| corr(mid_std, \|p\|) | +0.147 | 同上 |
| corr(n_tx, \|y\|) | +0.189 | — |
| corr(n_tx, \|p\|) | +0.195 | 已吸收 (两者相等) |
| corr(depth1, \|y\|) | **−0.189** | 薄盘→大波动 (负) |
| corr(depth1, \|p\|) | **+0.213** | 🔍 **符号学反了!** |
| corr(mid_range, y) | −0.014 | ✅ 确认 market 无方向信息 |

**分月稳定性**: corr(mr,|y|) 在 m25-70 全部 0.27~0.46 稳定; corr(mr,|p|) 仅 0.03~0.17 → 遗漏是**系统性**的, 非某月偶然。

### 裁决: ✅ P7-AMP 假设成立, 且证据比 GPT 预期的更强
- **幅度遗漏差距**: mid_range 有 0.371−0.105 = **0.27 的未利用幅度信息**
- **🔍 新发现 (GPT 未提)**: `depth1` 方向学反了 — baseline 给深盘样本更高 |p| (corr +0.213), 但实际深盘波动更小 (corr −0.19)。方向信号与幅度信号冲突时, 模型选了方向侧 → 幅度维度存在系统性误校准
- corr(|p|,|y|)=0.068 说明整个幅度维度基本未校准, 不是单特征问题

## 二、GPT 判断逐条裁决

| # | GPT 论断 | 裁决 | 依据 |
|---|---|---|---|
| 1 | 问题 = "哪些信息没被 152 压缩且能改变方向/权重" | ✅ | 与实测一致, 框架修正正确 |
| 2 | Market = sample importance / amplitude state, 非方向 alpha | ✅ | corr(mid_range,y)=−0.014 证实无方向; \|y\| 侧 0.37 证实有幅度 |
| 3 | SCFI +0.0040 是 representation geometry 而非独立 alpha | ✅ 推理合理 | LGB/73raw 只有 +0.000x 而 RealMLP +0.0040 → 架构依赖 |
| 4 | 优先级: mid_range > aggressiveness > 10/120s | ✅ | 审计实测支持 mid_range 第一 |
| 5 | P7-AMP 预期 +0.0005~+0.0012, +0.0015 是 upside | ⚠️ 参考 | 审计显示遗漏差距大 (0.27), 预期可能偏保守, 但门禁纪律不变 |
| 6 | 152 无 order→tx lag 特征 (机会 B) | ✅ | 列名核查: 仅有 t_sec_autocorr_lag1/5 (成交自身自相关), 无跨表 lag |
| 7 | 152 无 micro-batch topology (机会 C) | ✅ | 列名核查: 无 batch/micro 类特征 |
| 8 | 10s > 120s, 120s 低期望 | ✅ | 30/60/180 已夹住, 合理 |
| 9 | RED 路径: 关闭 market amplitude 路线转 event-time cross-modal | ✅ | 与审计的 P7-AMP 判决逻辑一致 |

## 三、P7-AMP Protocol 审计 (GPT 设计质量)

✅ **合格, 符合本地纪律**:
- 冻结 baseline 0.1351, 不允许重调 ✅
- 四臂 A/B/C/D 区分 "feature 增量" vs "amplitude gate 增量" ✅
- amplitude model 用 shallow LGB (depth≤3), 不预测方向 ✅
- α 只在 calibration split 选一次, 网格 0/0.25/0.5/0.75/1.0 ✅
- 门禁: GREEN ≥+0.0015 / YELLOW +0.0005~0.0015 / RED <+0.0005 ✅ 与本地量纲对齐
- 0 价格哨兵过滤进 max/min ✅

⚠️ **一处需修正**: GPT 的 arm B = "152 + mid_range600 → RealMLP" 直接加 feature。按本地 P5-B 经验 (73 raw 特征有效但小), arm B 预期低; 但作为对照保留无害。**真正的主结果应是 arm D vs C** (gate 含 mid_range 的独立价值)。

🔍 **一处补充 (基于实测新发现)**: depth1 学反 → gate 的输入特征列表应加入 `depth1` (GPT 已列) 且审计显示它可能贡献**负向校准修正** (把深盘样本的权重降下来)。这增加了 gate 的上行空间。

## 四、裁决结论

1. **GPT1 本轮质量显著高于上轮**: 框架修正 (Market=amplitude state) 被实测完全证实, P7-AMP protocol 设计合规
2. **P7-AMP 是当前证据最强的实验**: 幅度遗漏 0.27, |p|→|y| 仅 0.068, depth1 学反 — 三重证据指向同一结论: **cosine 下最大的未利用信息在幅度维度**
3. **执行顺序建议**: P7-AMP (P0) → order→tx lag response (P1.5) → aggressiveness (P1) → 10s (P2)。micro-batch (P3) 最弱, 可等 lag 结果再定
4. **不改变主线判断**: 若 P7-AMP RED, 关闭 market amplitude 路线, 转 event-time cross-modal (与 GPT 一致)

## 五、待用户拍板

- [ ] P7-AMP 全协议执行 (冻结 baseline → 5 amplitude 特征 → shallow LGB gate → α calibration → frozen 评估, ~1-2h)
- [ ] 或先只跑 arm A/C/D 快速版 (30min 出 Δ)
- [ ] 或与 order→tx lag 特征构建并行

## 复现

```bash
cd /d/mscapital-kaggle && export PYTHONPATH= && ./.venv/Scripts/python.exe scripts/p7amp_audit.py
```
(审计 ~1min, 输出 output/p7amp_audit.parquet)
