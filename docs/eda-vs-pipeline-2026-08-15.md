# EDA 发现 × 特征管线对照 (2026-08-15)

> 目的: 把 `docs/eda-raw-2026-08-15.md` 的发现与生产特征管线逐条对照, 定位真实缺口
> 管线: f0726 152 特征 (v7 主力, `reference/rfmf_0726data_source.py`) + 22 微观特征 (v5) + p5b 73 raw 聚合 (P5-D 验证有效)
> 验证方法: 200k 样本采样聚合, spearman 相关

## 对照表

| # | EDA 发现 | 强度 | 管线覆盖 | 判定 |
|---|---|---|---|---|
| 1 | mid_range(600s) vs \|t\| | +0.377 | f0726 只有 **m_mid_range_60** (60s 版), 全窗口版被 drop | ⚠️ **缺口 A** |
| 2 | mid_std(600s) vs \|t\| | +0.349 | m_mid_std (全窗口) ✅ | 已覆盖 |
| 3 | 卖压方向信号 (sell_ratio / tx_sell_ratio) | −0.054 / −0.061 | t_buy_ratio, o_buy_ratio_{15,30,45}, x_trans_order_buy_diff ✅ | 已覆盖 |
| 4 | 活动度→幅度 (n_order / n_tx) | +0.204 / +0.191 | o_n_{15,30,45}, o_sec_row_count, t_sec_* ✅ | 已覆盖 |
| 5 | 事件 60s 均匀 (无时变密度) | — | o/t_sec_*_near_far_ratio 已建模近/远比 | 已覆盖 (模型可学到≈1) |
| 6 | 订单买偏 + 成交卖偏不对称 | — | x_trans_order_buy_diff ✅ 显式建模 | 已覆盖 |
| 7 | imb1_last 方向信号 | +0.044 | m_imb_last ✅ | 已覆盖 |
| 8 | L2 深度不平衡偏买 | imb2 p50=0.108 | 管线只用 L1 imb | **验证: 非缺口** (见下) |
| 9 | 0 价格哨兵 (ask1=0) | 0.417% 行 | 无处理, 且 **100% 污染 mid** (mid=bid/2) | ⚠️ **缺口 B** (卫生) |
| 10 | depth→幅度 | −0.19 | m_*_weighted / o_sec_depth ✅ | 已覆盖 |

## 两个真实缺口

### 缺口 A: 600s mid_range 特征缺失
- EDA: 全窗口 mid_range vs |target| = **+0.377** (最强幅度信号)
- f0726 源码第 809 行构建了 `m_mid_range` (全窗口), 但最终特征表**只有 m_mid_range_60** —— 全窗口版被过滤/drop
- m_mid_std (0.349) 仍保留, 与 range 相关但 range 捕捉极值尾部, 可能有增量
- **低成本验证**: 补一列全窗口 mid_range → frozen PSEUDO 对照 (~30min)

### 缺口 B: 0 价格哨兵污染 mid 类特征
- ask1==0 时 bid_price_1 > 0 占 **100%** → mid = (0+bid)/2 = bid/2 ≈ 0.5, 严重失真
- 占比 0.417%, 对 mean/std 影响小, 但对 **m_mid_last / m_sp_last / m_imb_last** (last 类) 有定点污染 (0.4% 样本的末值错误)
- **低成本修复**: 预处理时 0 价格行排除出 mid/spread/imb 计算 (last 用最近有效行)

## 验证排除: L2 特征 (不做)
采样 200k 样本实测:
- imb1_last vs target = **+0.044** (管线已覆盖), imb2_last = +0.009 (弱 5×)
- corr(imb1_mean, imb2_mean) = 0.51 → 与 L1 信息重叠一半
- depth2_mean (−0.196) ≈ depth1_mean (−0.191) → 完全重叠
- l1_l2_gap 无信号 (−0.013)
→ **L2 深度不平衡不携带 L1 之外的独立信号, 不做 L2 特征族** (避免无效工程)

## 建议下一步 (按性价比)
1. **缺口 A 验证**: 全窗口 mid_range 补列 → frozen PSEUDO (30min, 门禁 +0.0015)
2. **缺口 B 修复**: 0 价格哨兵处理进预处理管线 (卫生, 预期小但免费)
3. 若 A 无增量: EDA 信号已全部被管线覆盖, 原始数据层信息挖掘收敛, 转向序列建模 (P5-02I 主线)

## 复现
```bash
cd /d/mscapital-kaggle && export PYTHONPATH= && ./.venv/Scripts/python.exe scripts/eda_raw_data.py  # 全量 EDA
# 缺口验证 (200k 采样, ~2min): 见本报告 §验证排除 的内联 polars 代码
```
