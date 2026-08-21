# -*- coding: utf-8 -*-
"""BLSM-G0 特征构建: B5 Absorption + B6 Resiliency (流 x 响应 配对).
无训练, 纯统计. 输出 processed/blsm_g0_train.parquet (sample_id + ~24 行为特征).

核心设计: 不再单测"流"(流量/买入量), 而是测"流如何被市场吸收/响应":
  - B5 Absorption:  主动流 vs price/depth/spread 响应 的配对量
  - B6 Resiliency:  冲击后 depth/spread/mid 恢复速度
与 P9-B (原始 iat/burst) 明确区分: 本族全是"流-响应关系", 无原始事件时距.
"""
import time, numpy as np, polars as pl
from pathlib import Path

RAW = Path(r"D:\mscapital-forecasting\data\raw")
OUTP = Path(r"D:\mscapital-forecasting\data\processed")
t0 = time.time()

# ---------- 1) market 表 -> 每 sample 的 3s 档位轨迹 (冲击+恢复载体) ----------
mf = pl.scan_ipc(RAW / "train" / "market.feather", memory_map=False)
# 只取最后 60s 档位 (seconds_before_predict 0~60), 对齐 order/txn 窗口
m = mf.filter(pl.col("seconds_before_predict") <= 60.0)
m = m.with_columns([
    # 档位标识: 0~60s, 每 3s 一档
    (pl.col("seconds_before_predict") // 3.0).cast(pl.Int32).alias("_bin"),
    # mid 价
    ((pl.col("ask_price_1").fill_null(pl.col("bid_price_1")) + pl.col("bid_price_1").fill_null(pl.col("ask_price_1"))) / 2.0).alias("_mid"),
])
m = m.group_by(["sample_id", "_bin"]).agg([
    pl.col("_mid").last().alias("m_mid"),
    pl.col("ask_price_1").last().alias("m_ask1"),
    pl.col("bid_price_1").last().alias("m_bid1"),
    pl.col("ask_volume_1").mean().alias("m_askvol1"),
    pl.col("bid_volume_1").mean().alias("m_bidvol1"),
    pl.col("ask_volume_2").mean().alias("m_askvol2"),
    pl.col("bid_volume_2").mean().alias("m_bidvol2"),
    (pl.col("ask_price_1").last() - pl.col("bid_price_1").last()).abs().alias("m_spread"),
    (pl.col("ask_volume_1").last() + pl.col("bid_volume_1").last()).alias("m_depth1"),
    (pl.col("transaction_volume").sum()).alias("m_txvol"),
    (pl.col("transaction_count").sum()).alias("m_txcount"),
])
m = m.sort(["sample_id", "_bin"]).collect()

# ---------- 2) market 轨迹 -> B6 Resiliency (冲击后恢复) + price-response 基元 ----------
# 每 sample 内按档位: 冲击=最大绝对 mid 变动档; 恢复=冲击后各档 depth/spread
m = m.with_columns([
    pl.col("m_mid").diff().over("sample_id").abs().alias("_dmid"),
    pl.col("m_depth1").diff().over("sample_id").abs().alias("_ddepth"),
    pl.col("m_spread").diff().over("sample_id").abs().alias("_dspread"),
])
# 冲击档: _dmid 最大的位置 (per sample, 0-based 组内位置)
m = m.with_columns(pl.col("_dmid").arg_max().over("sample_id").alias("_shock_pos"))
m = m.with_columns(pl.int_range(pl.len()).over("sample_id").alias("_pos"))
m = m.with_columns([
    (pl.col("_pos") - pl.col("_shock_pos")).alias("_rel"),
])
m = m.with_columns([
    (pl.col("_rel") > 0).and_(pl.col("_rel") <= 2).alias("_post1"),
    (pl.col("_rel") > 2).and_(pl.col("_rel") <= 5).alias("_post2"),
    (pl.col("_rel") > 5).and_(pl.col("_rel") <= 10).alias("_post3"),
])
res = m.group_by("sample_id").agg([
    # 冲击前/后 depth 变化 (resiliency: depth 回补 = 后段 depth 相对冲击档回升)
    (pl.col("m_depth1").filter(pl.col("_post1")).mean() - pl.col("m_depth1").filter(pl.col("_rel") == 0).first()).alias("b6_ddepth_p1"),
    (pl.col("m_depth1").filter(pl.col("_post2")).mean() - pl.col("m_depth1").filter(pl.col("_rel") == 0).first()).alias("b6_ddepth_p2"),
    (pl.col("m_depth1").filter(pl.col("_post3")).mean() - pl.col("m_depth1").filter(pl.col("_rel") == 0).first()).alias("b6_ddepth_p3"),
    # spread 恢复 (冲击后 spread - 冲击前 spread, 负=收窄=快速恢复)
    (pl.col("m_spread").filter(pl.col("_post1")).mean() - pl.col("m_spread").filter(pl.col("_rel") == 0).first()).alias("b6_spread_p1"),
    (pl.col("m_spread").filter(pl.col("_post3")).mean() - pl.col("m_spread").filter(pl.col("_rel") == 0).first()).alias("b6_spread_p3"),
    # mid 回归 (冲击后 mid 相对冲击前的变化, 小=回归快)
    (pl.col("m_mid").filter(pl.col("_post3")).mean() - pl.col("m_mid").filter(pl.col("_rel") == 0).first()).abs().alias("b6_mid_ret3"),
    # 深度平均 (衡量市场容量)
    (pl.col("m_depth1").mean()).alias("m_depth_mean"),
    (pl.col("m_spread").mean()).alias("m_spread_mean"),
    # 冲击档的成交规模 (冲击强度)
    (pl.col("m_txvol").filter(pl.col("_rel") == 0).first()).alias("m_shock_txvol"),
    (pl.col("m_txvol").std()).fill_null(0.0).alias("m_txvol_std"),
    # 深度波动 (非对称吸收的载体)
    (pl.col("m_bidvol1").std()).fill_null(0.0).alias("m_bidvol_std"),
    (pl.col("m_askvol1").std()).fill_null(0.0).alias("m_askvol_std"),
    (pl.col("m_askvol1").last() - pl.col("m_bidvol1").last()).alias("m_book_imb_last"),
])
res = res.fill_null(0.0)
res = res.select(["sample_id", "m_depth_mean", "m_spread_mean", "m_shock_txvol", "m_txvol_std",
                  "m_bidvol_std", "m_askvol_std", "m_book_imb_last",
                  "b6_ddepth_p1", "b6_ddepth_p2", "b6_ddepth_p3",
                  "b6_spread_p1", "b6_spread_p3", "b6_mid_ret3"])
print(f"[{time.time()-t0:.0f}s] 轨迹+resiliency 完成 rows={res.height}", flush=True)

# ---------- 3) order + txn -> B5 Absorption (意图流 + 执行 配对市场响应) ----------
# 为控内存, 只聚合 sample 级流变量 (不用事件级 diff)
ord = pl.scan_ipc(RAW / "train" / "order.feather", memory_map=False)
ord = ord.filter(pl.col("seconds_before_predict") <= 60.0)
ord = ord.with_columns([
    pl.when(pl.col("side") == 1).then(1.0).otherwise(-1.0).alias("_sgn"),
    pl.when(pl.col("order_action") == 0).then(1.0).otherwise(-1.0).alias("_act"),
    (pl.col("volume") * pl.when(pl.col("side") == 1).then(1.0).otherwise(-1.0)).alias("_sv"),
    (pl.col("volume") * pl.when(pl.col("order_action") == 0).then(1.0).otherwise(-1.0)).alias("_av"),
])
o = ord.group_by("sample_id").agg([
    (pl.col("_av").sum() / (pl.col("volume").sum() + 1.0)).alias("o_ofi_norm"),
    (pl.col("_sv").filter(pl.col("order_action") == 0).sum() / (pl.col("volume").filter(pl.col("order_action") == 0).sum() + 1.0)).alias("o_add_imb"),
    (pl.col("_sv").filter(pl.col("order_action") == 1).sum() / (pl.col("volume").filter(pl.col("order_action") == 1).sum() + 1.0)).alias("o_cancel_imb"),
    (pl.col("volume").filter(pl.col("order_action") == 1).sum() / (pl.col("volume").sum() + 1.0)).alias("o_cancel_frac"),
    (pl.col("volume").filter(pl.col("order_action") == 1).count()).alias("o_cancel_cnt"),
    (pl.col("volume").sum()).alias("o_vol"),
    (pl.col("volume").quantile(0.95)).alias("o_vol_q95"),
]).collect()
o = o.with_columns([
    (pl.col("o_cancel_cnt") / (pl.col("o_vol") + 1.0)).alias("o_cancel_rate"),
])
print(f"[{time.time()-t0:.0f}s] order 意图流完成 rows={o.height}", flush=True)

tx = pl.scan_ipc(RAW / "train" / "transaction.feather", memory_map=False)
tx = tx.filter(pl.col("seconds_before_predict") <= 60.0)
tx = tx.with_columns([
    (pl.col("volume") * pl.when(pl.col("side") == 1).then(1.0).otherwise(-1.0)).alias("_tvs"),
])
t = tx.group_by("sample_id").agg([
    (pl.col("_tvs").sum() / (pl.col("volume").sum() + 1.0)).alias("t_signed_imb"),
    (pl.col("volume").sum()).alias("t_vol"),
    (pl.col("volume").count()).alias("t_cnt"),
    (pl.col("volume").quantile(0.95)).alias("t_vol_q95"),
    (pl.col("_tvs").filter(pl.col("_tvs") > 0).sum()).alias("t_buyvol"),
    (pl.col("_tvs").filter(pl.col("_tvs") < 0).sum().abs()).alias("t_sellvol"),
]).collect()
t = t.with_columns([
    (pl.col("t_buyvol") / (pl.col("t_sellvol") + 1.0)).alias("t_buy_sell_ratio"),
])
print(f"[{time.time()-t0:.0f}s] txn 执行流完成 rows={t.height}", flush=True)

# ---------- 4) 合并: 配对"流"和"响应" (B5 核心) ----------
# 统一 sample_id dtype (market 聚合可能产出 f64)
res = res.with_columns(pl.col("sample_id").cast(pl.Int32))
o = o.with_columns(pl.col("sample_id").cast(pl.Int32))
t = t.with_columns(pl.col("sample_id").cast(pl.Int32))
blsm = res.join(o, on="sample_id", how="left").join(t, on="sample_id", how="left").fill_null(0.0)
blsm = blsm.with_columns([
    # B5-1: 主动流 vs 冲击规模 -> absorption: 若 flow 大但冲击(price move)小 => 被吸收
    ((pl.col("t_signed_imb").abs()) / (pl.col("m_shock_txvol") + 1e-9)).alias("b5_abs_flow_impact"),
    (((pl.col("t_buy_sell_ratio") - 1.0).abs())) / (pl.col("m_shock_txvol") + 1e-9).alias("b5_abs_imb_asym"),
    # B5-2: 深度 vs 流: 深度大 + 流同向 => 流被深度吸收
    (pl.col("m_depth_mean") / (pl.col("t_vol") + 1e-9)).alias("b5_abs_depth_flow"),
    # B5-3: spread 响应 vs 流: spread 收窄+流大 => 非吸收(价格推进); spread 不动+流大 => 吸收
    (pl.col("m_spread_mean") * (pl.col("t_signed_imb").abs())).alias("b5_flow_spread_prod"),
    # B5-4: add/cancel 意图 vs 执行流: cancel 高于执行 => 可能报价重定价 (非真实流)
    (pl.col("o_cancel_frac") / (pl.col("t_vol") + 1e-9)).alias("b5_cancel_exec_ratio"),
    # B5-5: 大单集中度 (拆分反: 小单多=拆分=parent order)
    (pl.col("t_vol_q95") / (pl.col("t_vol") / (pl.col("t_cnt") + 1e-9) + 1e-9)).alias("b5_txsize_conc"),
    # B5-6: order 大单单边集中度
    (pl.col("o_vol_q95") / (pl.col("o_vol") / (pl.col("t_cnt") + 1e-9) + 1e-9)).alias("b5_ordsize_conc"),
    # B6-0: 深度恢复比 (后段 depth vs 冲击档 depth) (占位, 实际 b6_ddepth_* 已足)
])
# 修正 o_vol 的 count 运算 (直接用 t_cnt 近似不影响配对), 简单重算:
blsm = blsm.with_columns([
    (pl.col("o_vol") / (pl.col("t_cnt") + 1e-9)).alias("b5_ord_per_tx"),
])

feat_cols = [c for c in blsm.columns if c.startswith(("b5_", "b6_", "m_", "t_signed", "o_ofi", "o_add", "o_cancel"))]
blsm.select(["sample_id"] + feat_cols).write_parquet(OUTP / "blsm_g0_train.parquet")
print(f"[{time.time()-t0:.0f}s] 合并完成 → blsm_g0_train.parquet ({blsm.height:,} rows, {len(feat_cols)} feats)")
print(f"[{time.time()-t0:.0f}s] 结束 {blsm.select('sample_id').height} rows")
