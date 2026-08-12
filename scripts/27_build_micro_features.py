# -*- coding: utf-8 -*-
"""
P1-1a: Stationary Microstructure Representation 构建
从 order/transaction 原始数据构建无量纲 primitive 特征 (F1-F6 组)
输出: processed/micro_features_{train,test}.parquet (sample_id + 新特征)
"""
import time
import polars as pl

RAW = r"D:\mscapital-forecasting\data\raw"
OUT = r"D:\mscapital-forecasting\data\processed"

def build_order_feats(split):
    """order 表 (60s 原始订单流): add/cancel imbalance, 到达率, burstiness, 大单"""
    lf = pl.scan_ipc(f"{RAW}/{split}/order.feather", memory_map=False)
    lf = lf.sort(["sample_id", "seconds_before_predict"], descending=[False, True])
    lf = lf.with_columns([
        pl.when(pl.col("side") == 0).then(1.0).otherwise(-1.0).alias("_sgn"),
        pl.when(pl.col("order_action") == 0).then(1.0).otherwise(-1.0).alias("_act"),
        (pl.col("volume") * pl.when(pl.col("side") == 0).then(1.0).otherwise(-1.0)).alias("_sv"),
        (pl.col("volume") * pl.when(pl.col("order_action") == 0).then(1.0).otherwise(-1.0)).alias("_av"),
        # 事件到达间隔 (排序后 diff)
        pl.col("seconds_before_predict").diff().over("sample_id").abs().alias("_iat"),
        # 局部大单阈值: 组内 90 分位
        pl.col("volume").quantile(0.90).over("sample_id").alias("_q90"),
    ])
    exprs = [
        # F2: add/cancel imbalance
        ((pl.col("_sv").filter(pl.col("order_action") == 0)).sum()
         / (pl.col("volume").filter(pl.col("order_action") == 0).sum() + 1.0)).alias("o_add_imb"),
        ((pl.col("_sv").filter(pl.col("order_action") == 1)).sum()
         / (pl.col("volume").filter(pl.col("order_action") == 1).sum() + 1.0)).alias("o_cancel_imb"),
        (pl.col("volume").filter(pl.col("order_action") == 0).sum()
         / (pl.col("volume").filter(pl.col("order_action") == 1).sum() + 1.0)).alias("o_add_cancel_ratio"),
        # F1: 订单流 OFI (action 加权归一化)
        (pl.col("_av").sum() / (pl.col("volume").sum() + 1.0)).alias("o_ofi_norm"),
        # F3: 到达率 + burstiness
        (pl.col("_iat").count() / 60.0).alias("o_arrival_rate"),
        (pl.col("_iat").mean().log1p()).alias("o_iat_mean_log"),
        (pl.col("_iat").std() / (pl.col("_iat").mean() + 1e-8)).alias("o_iat_cv"),
        (pl.col("_iat").max() / (pl.col("_iat").median() + 1e-8)).alias("o_burst_ratio"),
        # F4: 大单相对不平衡 (局部 q90)
        ((pl.col("_sv").filter(pl.col("volume") > pl.col("_q90"))).sum()
         / (pl.col("volume").filter(pl.col("volume") > pl.col("_q90")).sum() + 1.0)).alias("o_large_imb"),
        (pl.col("volume").filter(pl.col("volume") > pl.col("_q90")).sum()
         / (pl.col("volume").sum() + 1.0)).alias("o_large_share"),
    ]
    return lf.group_by("sample_id").agg(exprs).collect(streaming=True)

def build_tx_feats(split):
    """transaction 表 (60s 成交流): 强度, burstiness, 大单, fast-slow flow"""
    lf = pl.scan_ipc(f"{RAW}/{split}/transaction.feather", memory_map=False)
    lf = lf.sort(["sample_id", "seconds_before_predict"], descending=[False, True])
    lf = lf.with_columns([
        pl.when(pl.col("side") == 0).then(1.0).otherwise(-1.0).alias("_sgn"),
        (pl.col("volume") * pl.when(pl.col("side") == 0).then(1.0).otherwise(-1.0)).alias("_sv"),
        pl.col("seconds_before_predict").diff().over("sample_id").abs().alias("_iat"),
        pl.col("volume").quantile(0.90).over("sample_id").alias("_q90"),
    ])
    exprs = [
        # F5: impact 代理 (签名量占比)
        (pl.col("_sv").sum() / (pl.col("volume").sum() + 1.0)).alias("t_signed_norm"),
        # F3: 成交强度 + burstiness
        (pl.col("_iat").count() / 60.0).alias("t_arrival_rate"),
        (pl.col("_iat").std() / (pl.col("_iat").mean() + 1e-8)).alias("t_iat_cv"),
        # F4: 大单
        ((pl.col("_sv").filter(pl.col("volume") > pl.col("_q90"))).sum()
         / (pl.col("volume").filter(pl.col("volume") > pl.col("_q90")).sum() + 1.0)).alias("t_large_imb"),
        (pl.col("volume").filter(pl.col("volume") > pl.col("_q90")).sum()
         / (pl.col("volume").sum() + 1.0)).alias("t_large_share"),
        # F6: fast(5s) vs slow(60s) flow 分歧
        ((pl.col("_sv").filter(pl.col("seconds_before_predict") <= 5)).sum()
         / (pl.col("volume").filter(pl.col("seconds_before_predict") <= 5).sum() + 1.0)
         - (pl.col("_sv").sum() / (pl.col("volume").sum() + 1.0))).alias("t_flow_fast_slow"),
    ]
    return lf.group_by("sample_id").agg(exprs).collect(streaming=True)

def build_market_feats(split):
    """market 表补充: microprice gap (无量纲)"""
    lf = pl.scan_ipc(f"{RAW}/{split}/market.feather", memory_map=False)
    lf = lf.sort(["sample_id", "seconds_before_predict"], descending=[False, True])
    lf = lf.with_columns([
        ((pl.col("ask_price_1") + pl.col("bid_price_1")) * 0.5).alias("_mid"),
        (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("_sp"),
        # microprice: 成交量加权价
        ((pl.col("ask_price_1") * pl.col("bid_volume_1") + pl.col("bid_price_1") * pl.col("ask_volume_1"))
         / (pl.col("ask_volume_1") + pl.col("bid_volume_1") + 1.0)).alias("_micro"),
    ])
    exprs = [
        (((pl.col("_micro") - pl.col("_mid")) / (pl.col("_sp") + 1e-8)).last()).alias("m_micro_gap_last"),
        (((pl.col("_micro") - pl.col("_mid")) / (pl.col("_sp") + 1e-8)).mean()).alias("m_micro_gap_mean"),
        # 相对价差 (mid 归一)
        (pl.col("_sp").last() / (pl.col("_mid").last() + 1e-8)).alias("m_sp_rel_last"),
        (pl.col("_sp").mean() / (pl.col("_mid").mean() + 1e-8)).alias("m_sp_rel_mean"),
        # 深度不平衡 L2 版
        ((pl.col("ask_volume_2") - pl.col("bid_volume_2"))
         / (pl.col("ask_volume_2") + pl.col("bid_volume_2") + 1.0)).last().alias("m_imb2_last"),
        ((pl.col("ask_volume_2") - pl.col("bid_volume_2"))
         / (pl.col("ask_volume_2") + pl.col("bid_volume_2") + 1.0)).mean().alias("m_imb2_mean"),
    ]
    return lf.group_by("sample_id").agg(exprs).collect(streaming=True)

T0 = time.time()
for split in ["train", "test"]:
    t0 = time.time()
    of = build_order_feats(split)
    print(f"{split} order: {of.shape} ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    tf = build_tx_feats(split)
    print(f"{split} tx: {tf.shape} ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    mf = build_market_feats(split)
    print(f"{split} market: {mf.shape} ({time.time()-t0:.0f}s)", flush=True)
    df = (of.rename({"sample_id": "sid"})
          .join(tf.rename({"sample_id": "sid"}), on="sid", how="full", coalesce=True)
          .join(mf.rename({"sample_id": "sid"}), on="sid", how="full", coalesce=True)
          .rename({"sid": "sample_id"}))
    df.write_parquet(f"{OUT}/micro_features_{split}.parquet")
    print(f"{split} saved: {df.shape} cols={[c for c in df.columns if c != 'sample_id']}", flush=True)
print(f"TOTAL {time.time()-T0:.0f}s", flush=True)
