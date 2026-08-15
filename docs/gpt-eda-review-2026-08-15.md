# GPT EDA 评审 × 本地实测裁决 (2026-08-15)

> 输入: GPT 结构级探索 (基于 Kaggle 官方页描述, 无本地数值)
> 方法: 与本地全量实测 (`docs/eda-raw-2026-08-15.md`, `docs/data-generation-structure.md`, `docs/eda-vs-pipeline-2026-08-15.md`, RESULTS.md 台账) 逐条对照
> 裁决符号: ✅ 一致/已做 / ⚠️ 部分 / ❌ 修正 / 🔍 新发现

## 一、框架层 (GPT 说对的部分)

| # | GPT 论断 | 本地实测 | 裁决 |
|---|---|---|---|
| 1 | 三信息层级 Market State → Order Intent → Trade Realization | 数据生成结构取证 (zz_forensics_datagen) 确认 600s 压缩层 + 60s 明细层 | ✅ 完全一致 |
| 2 | market = 等间隔 bar, order/tx = raw event flow | 实测 bar 间隔 p50=3.00s, order/tx 连续时间戳 | ✅ |
| 3 | side: 0=Buy 1=Sell; order_action: 0=New 1=Cancel | 实测一致 (75.3% new / 24.7% cancel) | ✅ |
| 4 | cosine 指标: scale 不重要, 方向+相对幅度重要 | 更深入: MSE 失败机制 (target=sign·\|y\|, E[sign\|x\]≈0), 幅度是富矿 (corr 0.43-0.47, 大波动 AUC 0.78) | ✅ 一致且本地更深 |
| 5 | Buy/Sell × New/Cancel 四象限拆开 | **P5-B 73 raw 特征已做** (ob_bid_add_cnt 等), 生产级验证 +0.00067 | ✅ 已做已验证 |
| 6 | intent-realization divergence (order 强买但 trade 没跟上) | `x_trans_order_buy_diff` 已建模; EDA 实测订单买偏 51.2% + 成交卖偏 51.7% 的不对称 | ✅ 已覆盖 |
| 7 | O_innovation = O_actual − E[O\|M] 条件化 | **= P5-B SCFI 的 Z 特征** (事件流 − E[·\|M]): RealMLP +0.0040, LGB3seed +0.00085 已验证 | ✅ GPT 独立提出同一思路, 本地已实现 |
| 8 | Market 作为条件变量 (regime-conditioned alpha) | E01/E02/P3-02/P6R 全验证: **条件化残差 5 连杀** (P6R KILL), 条件化特征 (Z) 有效 | ⚠️ 方向正确但结论需细分 |
| 9 | 10 步 EDA 清单 | 全部已在本地执行过 (label/rows-per-sample/多尺度/四象限/cross-modal/drift/metric-aligned frozen cosine) | ✅ 阶段已超越 |

## 二、GPT 不知道的实测事实 (❌ 需修正的判断)

| # | GPT 判断 | 实测修正 |
|---|---|---|
| 1 | "market 是最重要的一张表" | **半对**: 34 个 600s 聚合特征 PSEUDO −0.0002 **无信号**; 但 600s 序列 + cosine 有信号 (corr 0.086) → market 的价值在**序列形式**, 不在统计量形式 |
| 2 | "最看好的表示是 conditional innovation" | 已实测: Z 特征增量真实但**有限** (单 learner +0.0040, 生产 blend 后收窄), 不是突破级 |
| 3 | "Order 四象限非常值钱" | 已实测: side×action 拆分 +0.00067 (生产验证), 真实但小 |
| 4 | 隐含假设 "信息在低维统计量里" | P3 系列全灭: SAE/TinyLOBERT/2.5D 网格全部被 152 特征覆盖 → **信息已不在低维表示, 在序列本身** |

## 三、GPT 提出的本地尚未覆盖的点 🔍 (新增候选)

### 候选 1: Order price aggressiveness (订单价格距 mid 距离)
- GPT 建议 "订单价格距离当前 mid 的距离"
- 本地核查: f0726 只有 `x_sec_price_mid_diff` (秒级聚合版), **原始 order 事件的 price−mid 距离特征缺失**
- EDA 背景: order price p1=0.9715 / p99=1.0510 (mid ±5%), 深挂单极值 0.0002~8.63
- 预期: 可能是小增量 (秒级版已覆盖部分), 但**未验证**

### 候选 2: 多尺度网格缺口 (10s / 120s)
- GPT 建议 10/30/60/120/300/600 多尺度 ΔX = X_recent − X_long
- 本地核查: 已有 30/60/180/300/600 (部分), **缺 10s 和 120s 档**; 已有 x_rv_60_180_ratio / x_m_vol_long_short_ratio 做长短比
- 预期: 边际增量 (P5-02I 已证短程 ≤10 步 = 30s 有信号, 但 152 特征族窗口已覆盖 15/30/45/60)

### 候选 3: 全窗口 mid_range (600s) — 与 GPT 的多尺度建议同源
- 本地已发现 (EDA vs Pipeline 缺口 A): 最强幅度信号 +0.377, f0726 全窗口版被 drop
- **优先级最高**: 30min frozen 验证

## 四、裁决结论

1. **GPT 框架层判断与本地全部一致** — 三信息层级、四象限、divergence、条件化思路, 本地均已实现并验证过 (多数在 GPT 之前)
2. **GPT 的"条件化创新"核心主张 = 已完成的 SCFI Z 实验** — 结论已实测: 有效但非突破
3. **新增候选仅 3 个**: 缺口 A (600s mid_range, 本地已发现) + aggressiveness 特征 + 10s/120s 档
4. **GPT 未做真实数值分析** (自己声明"结构级"), 其 10 步 EDA 清单本地全部执行完毕且走得更远 (P5-D/E 生产验证阶段)

## 五、行动建议 (按性价比, 待拍板)

| 优先级 | 动作 | 成本 | 预期 |
|---|---|---|---|
| 1 | 缺口 A: 600s mid_range 补列 → frozen PSEUDO | ~30min | +0.0015 门禁待验 |
| 2 | 候选 1: order price−mid aggressiveness 特征组 → 补进 73 raw 管线验证 | ~1h | 小增量, 未验证 |
| 3 | 候选 2: 10s/120s 多尺度补档 | ~1h | 边际, 低优先 |

**总体判断**: GPT EDA 是合格的结构级框架梳理, 但未提供本地未实测过的新信息源; 唯一实质新增是 aggressiveness 特征组。不改变当前主线 (P5-02I 序列建模), 候选实验可并入等待队列。
