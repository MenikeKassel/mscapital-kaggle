# P8-01A: O→T Temporal-Arrow Diagnostic — 负结果报告 (2026-08-15)

> 实验: `scripts/p8_01a_ot_lag.py` | 数据: `output/p8_01a_ot_lag/results.json`
> 协议: v2 方案 (docs/otlag-probe-plan-v2.md) Stage P8-01A, 无 ML 诊断
> 采样: 149,952 样本 (月分层 0-70), 1s bin × 61 bins, 20 reps placebos
> 判定: **Gate G0 FAIL → O→T lag response 主线 STOP**

---

## 1. 结果摘要 (15 万样本, 全 71 月)

### 1.1 响应曲线 (z-score 互相关, 全窗口 Pearson)

```
signed_new -> signed_tx:
lag:   -30    -10     -5     -3     -1      0     +1     +3     +5    +10    +30
z:   0.004  -0.002  0.004  0.015  0.026  0.328  -0.002 -0.001 -0.005 -0.004  0.006

buy_new -> buy_tx:
z:   0.008  -0.003  0.003  0.021  0.036  0.360   0.010  0.007 -0.005 -0.005  0.009

buy_cancel -> sell_tx:
z:   0.002  -0.003 -0.001  0.007  0.016  0.041   0.009  0.008  0.002 -0.001  0.002
```

**判读**: 唯一的峰在 **lag=0 (同期 z≈0.33-0.36)**; 所有非零 lag 的 z 都在 ±0.01 噪声量级。
**没有跨秒响应结构**——曲线平坦。

### 1.2 方向性 (A = forward − backward)

- A_raw 全部为负: C_raw(+1) 远小于 C_raw(−1) (signed: 897万 vs 3543万) — backward > forward
- **月度 A_z > 0: 0/71 个月** (A_z 均值 −0.0279) — 没有任何一个月支持 O→T 正向
- reverse placebo (T→O): z(+1)=0.0256 > O→T 的 −0.0015 — 若存在方向性, 也是 T→O 而非 O→T

### 1.3 Placebo 对照

| placebo | null z(+1) | 真实 z(+1) | 判读 |
|---|---|---|---|
| block shift (破坏 O-T 对齐) | −0.0053 ± 0.0007 | −0.0015 | 真实值在 null 同量级 (无超出) |
| marginal shuffle | −0.0002 ± 0.0008 | −0.0015 | shuffle 归零 ✓, 真实值 ≈ 0 |
| reverse (T→O) | +0.0256 | — | 反向略强, 但绝对值仍噪声级 |

### 1.4 活动度分层 (signed pair)

low: z(+1)=0.0003 (n=50,575) / mid: +0.0008 (n=50,819) / high: −0.0067 (n=48,558)
— 无隐藏的活动度依赖; 高活动层甚至为负。

## 2. 结论

**Gate G0 FAIL**: O→T 跨表时序响应在 1s bin 尺度上**不存在**。

- order 与 transaction 的关系是**同期机械耦合** (lag=0 z≈0.33): transaction 本质是 order 撮合结果的另一日志视图 (GPT2 预警的"机械重合"风险 5 实证确认);
- 该同期信息**已被 152 特征覆盖** (t_vol_sum/t_buy_ratio/o_vol_sum 等同期统计);
- 跨秒滞后结构 (lag=±1..±30s) 全为噪声, 71 个月零支持, placebo 全覆盖。

**边界声明** (诚实):
1. 结论限于 1s bin 尺度; 亚秒级响应不可测 (且 GPT2 认为毫秒级 event matching 无可靠经济含义), 且即使存在也只会落入 lag=0 的机械重合桶。
2. 本实验判死的是"跨表时序箭头"; 不否定"跨表同期统计" (152 已有) 与"market 表内跨通道同步" (P5-02I GREEN)。

## 3. 对相关方向的连带影响

| 方向 | 影响 | 处置 |
|---|---|---|
| O→T lag response (P8-01B/C, CrossStream) | **证伪** | 主线 STOP, 不开 Stage 2/3 |
| ETCI 事件时 (time-to-first-tx) | 连带受损: 若无 O→T 响应, "响应时延"大概率只是活动度特征 (GPT1 预警) | 降级为低优先, 如需测须带 hazard/background 对照 |
| lb142 v9 CrossStreamMixer 的独立信息归因 | 不再支持"O→T 时序"假说 | 归因转向: market 600s 原始序列 / 未知 grid 通道 / factors / ensemble diversity |
| T→O 方向 | z(+1)=0.026 略高但为噪声级, 无月度验证 | 不追 |

## 4. 复现路径

```
scripts/p8_01a_ot_lag.py --n-samples 150000 --reps 20   (56s, 无 GPU)
output/p8_01a_ot_lag/results.json                        (全量统计)
```

P0 leakage 单元测试内建 (窗口内对齐/非负/有限值断言) 已 PASS。
