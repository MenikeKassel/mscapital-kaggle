# -*- coding: utf-8 -*-
"""MSCapital 原始数据集 EDA v2 — 低内存分块版 (2026-08-15 重构后重跑)

对比 v1 (eda_raw_data.py):
- market 2.2 亿行改为按 sample_id 分块 (30 万/块) + lazy scan, 峰值内存 <2GB
  (系统有 p9/p10 训练在跑, 全量物化会 OOM)
- 吸收互联网 EDA (Pavlo Ivanin "Quantitative Market Microstructure EDA") 的先进概念,
  新增量化分析层: micro-price gap / CVD / spread 压缩 / L1 depth 演化 vs target 双轨 spearman

层次: 1) label  2) market 盘口+路径  3) order/tx 事件流  4) 跨表关联 (含新微观结构量)
      5) train vs test 漂移
输出: output/eda/stats_v2.txt
"""
import polars as pl
import numpy as np
import os, json, time

RAW = "D:/mscapital-forecasting/data/raw"
OUT = "D:/mscapital-kaggle/output/eda"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)
T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)

lines = []
def rec(msg):
    lines.append(str(msg))
    print(msg, flush=True)

CHUNK = 300_000  # 每块样本数
SMOKE = int(os.environ.get("MSCAP_EDA_SMOKE", "0"))
if SMOKE:
    CHUNK = 300_000
    log(f"SMOKE MODE: {SMOKE} samples")

# ============ 1) LABEL ============
log("loading label")
lab = pl.read_ipc(f"{RAW}/train/label.feather")
t = lab["target"]
rec("="*70)
rec("1) LABEL (n=%d)" % len(lab))
rec(f"   mean={t.mean():.3e} std={t.std():.3e} min={t.min():.4f} max={t.max():.4f}")
rec(f"   quantiles: " + " ".join(f"p{q*10}={t.quantile(q):.5f}" for q in [0.01,0.1,0.25,0.5,0.75,0.9,0.99]))
rec(f"   pos%={(t>0).mean()*100:.2f} zero%={(t==0).mean()*100:.4f}")
rec(f"   |t| quantiles: " + " ".join(f"p{q*10}={t.abs().quantile(q):.5f}" for q in [0.5,0.75,0.9,0.99,0.999]))
mstat = (lab.group_by("month").agg(
    pl.col("target").mean().alias("mean"),
    pl.col("target").std().alias("std"),
    pl.col("target").count().alias("n"),
    (pl.col("target") > 0).mean().alias("pos_ratio"),
    pl.col("target").abs().mean().alias("abs_mean"),
).sort("month"))
rec("\n   月度 target 演化 (每 10 月抽样):")
for r in mstat.iter_rows(named=True):
    if r["month"] % 10 == 0 or r["month"] == 70:
        rec(f"   m{r['month']:3d}: n={r['n']:6d} mean={r['mean']:+.2e} std={r['std']:.2e} pos={r['pos_ratio']:.3f} |t|={r['abs_mean']:.2e}")
rec(f"   |t|>2*std: {(t.abs()>2*t.std()).mean()*100:.2f}%   |t|>4*std: {(t.abs()>4*t.std()).mean()*100:.2f}%")
rec(f"   std 月度漂移: min={mstat['std'].min():.2e} max={mstat['std'].max():.2e} ratio={mstat['std'].max()/mstat['std'].min():.2f}")
rec(f"   mean 月度漂移: min={mstat['mean'].min():+.2e} max={mstat['mean'].max():+.2e}")

# ============ 2) MARKET 分块 ============
log("market: chunked scan (300k samples/chunk)")
sids = lab["sample_id"].to_numpy()
if SMOKE:
    sids = sids[:SMOKE]
n_chunks = int(np.ceil(len(sids) / CHUNK))
chunks = []
for i in range(n_chunks):
    chunk_ids = sids[i*CHUNK:(i+1)*CHUNK]
    c = (pl.scan_ipc(f"{RAW}/train/market.feather")
         .filter(pl.col("sample_id").is_in(chunk_ids))
         .with_columns([
             ((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2).cast(pl.Float32).alias("mid1"),
             (pl.col("ask_price_1") - pl.col("bid_price_1")).cast(pl.Float32).alias("spread1"),
             ((pl.col("bid_volume_1") - pl.col("ask_volume_1")) / (pl.col("bid_volume_1") + pl.col("ask_volume_1") + 1.0)).cast(pl.Float32).alias("imb1"),
             ((pl.col("bid_volume_2") - pl.col("ask_volume_2")) / (pl.col("bid_volume_2") + pl.col("ask_volume_2") + 1.0)).cast(pl.Float32).alias("imb2"),
             # micro-price (L1 流动性加权)
             ((pl.col("bid_volume_1") * pl.col("ask_price_1") + pl.col("ask_volume_1") * pl.col("bid_price_1"))
              / (pl.col("bid_volume_1") + pl.col("ask_volume_1") + 1.0)).cast(pl.Float32).alias("micro_p"),
         ])
         .group_by("sample_id")
         .agg([
             (pl.col("mid1").max() - pl.col("mid1").min()).alias("mid_range"),
             pl.col("mid1").std().alias("mid_std"),
             (pl.col("mid1").last() - pl.col("mid1").first()).alias("mid_drift"),
             pl.col("mid1").first().alias("mid_first"),
             (pl.col("mid1").last()).alias("mid_last"),
             (pl.col("micro_p") - pl.col("mid1")).mean().alias("micro_gap_mean"),
             (pl.col("micro_p") - pl.col("mid1")).last().alias("micro_gap_last"),
             (pl.col("micro_p") - pl.col("mid1")).std().alias("micro_gap_std"),
             pl.col("spread1").mean().alias("spread_mean"),
             pl.col("spread1").std().alias("spread_std"),
             (pl.col("spread1").last()).alias("spread_last"),
             # spread 压缩: 近期 vs 早期
             (pl.col("spread1").filter(pl.col("seconds_before_predict") < 30).mean()).alias("spread_near"),
             (pl.col("spread1").filter(pl.col("seconds_before_predict") >= 300).mean()).alias("spread_far"),
             pl.col("imb1").mean().alias("imb1_mean"),
             pl.col("imb1").last().alias("imb1_last"),
             pl.col("imb2").mean().alias("imb2_mean"),
             pl.col("imb2").last().alias("imb2_last"),
             # L1 depth 演化
             pl.col("bid_volume_1").last().alias("bidv1_last"),
             pl.col("ask_volume_1").last().alias("askv1_last"),
             pl.col("bid_volume_1").mean().alias("bidv1_mean"),
             pl.col("ask_volume_1").mean().alias("askv1_mean"),
             pl.col("bid_volume_1").std().alias("bidv1_std"),
             pl.col("ask_volume_1").std().alias("askv1_std"),
             pl.col("transaction_volume").sum().alias("bar_vol"),
             (pl.col("imb1") * pl.col("transaction_volume")).sum().alias("vwap_imb"),
         ])
         .collect(engine="streaming"))
    chunks.append(c)
    log(f"   chunk {i+1}/{n_chunks} done ({len(c)} samples)")
path = pl.concat(chunks)
# 汇总统计
rec("="*70)
rec("2) MARKET (每样本聚合, n=%d)" % len(path))
rec(f"   0价格哨兵 由 v1 覆盖 (ask1 0.41%)")
rec(f"   每样本 mid_range: p50={path['mid_range'].quantile(0.5):.2e} p90={path['mid_range'].quantile(0.9):.2e} p99={path['mid_range'].quantile(0.99):.2e}")
rec(f"   每样本 mid_std:   p50={path['mid_std'].quantile(0.5):.2e} p90={path['mid_std'].quantile(0.9):.2e}")
rec(f"   micro_gap mean:   p10={path['micro_gap_mean'].quantile(0.1):+.2e} p50={path['micro_gap_mean'].quantile(0.5):+.2e} p90={path['micro_gap_mean'].quantile(0.9):+.2e}")
rec(f"   micro_gap last:   p10={path['micro_gap_last'].quantile(0.1):+.2e} p50={path['micro_gap_last'].quantile(0.5):+.2e} p90={path['micro_gap_last'].quantile(0.9):+.2e}")
rec(f"   spread_mean:      p50={path['spread_mean'].quantile(0.5):.2e} p90={path['spread_mean'].quantile(0.9):.2e}")
rec(f"   spread 压缩比 (near30/far300): p50={path.select((pl.col('spread_near')/pl.col('spread_far').clip(lower_bound=1e-12)).quantile(0.5)).item():.3f}")
rec(f"   imb1_last:        p10={path['imb1_last'].quantile(0.1):+.3f} p50={path['imb1_last'].quantile(0.5):+.3f} p90={path['imb1_last'].quantile(0.9):+.3f}")
rec(f"   imb2_last:        p50={path['imb2_last'].quantile(0.5):+.3f}")
rec(f"   bidv1_last:       p50={path['bidv1_last'].quantile(0.5):.0f}  askv1_last: p50={path['askv1_last'].quantile(0.5):.0f}")
rec(f"   bidv1_std:        p50={path['bidv1_std'].quantile(0.5):.0f}  askv1_std: p50={path['askv1_std'].quantile(0.5):.0f}")

# 与 label 双轨关联
merged = path.join(lab.select(["sample_id", "target"]), on="sample_id", how="inner")
rec("\n   双轨 spearman (方向 vs |幅度|):")
for c in ["mid_range", "mid_std", "bar_vol", "micro_gap_mean", "micro_gap_last", "micro_gap_std",
          "spread_mean", "spread_std", "spread_last", "imb1_mean", "imb1_last", "imb2_mean", "imb2_last",
          "bidv1_last", "askv1_last", "bidv1_std", "askv1_std"]:
    r = merged.select(pl.corr(pl.col(c), pl.col("target"), method="spearman").alias("r")).item()
    ra = merged.select(pl.corr(pl.col(c), pl.col("target").abs(), method="spearman").alias("r")).item()
    rec(f"   {c:16s} vs target={r:+.4f}  vs |target|={ra:+.4f}")

# ============ 3) ORDER / TX (scan + streaming) ============
log("order: streaming agg")
ord_ = (pl.scan_ipc(f"{RAW}/train/order.feather")
        .group_by("sample_id").agg([
            pl.len().alias("n_order"),
            (pl.col("side") == 1).sum().alias("n_sell"),
            (pl.col("order_action") == 1).sum().alias("n_cancel"),
            pl.col("volume").sum().alias("vol_order"),
            pl.col("volume").mean().alias("vol_order_mean"),
        ]).collect(engine="streaming"))
rec("="*70)
rec("3) ORDER (每样本聚合, n=%d)" % len(ord_))
oc = ord_["n_order"]
rec(f"   每样本事件数: p10={oc.quantile(0.1):.0f} p50={oc.quantile(0.5):.0f} p90={oc.quantile(0.9):.0f} p99={oc.quantile(0.99):.0f} max={oc.max()}")

log("transaction: streaming agg")
txn = (pl.scan_ipc(f"{RAW}/train/transaction.feather")
       .group_by("sample_id").agg([
           pl.len().alias("n_tx"),
           (pl.col("side") == 1).sum().alias("n_tx_sell"),
           pl.col("volume").sum().alias("vol_tx"),
           pl.col("volume").mean().alias("vol_tx_mean"),
           # CVD 终值 = 累计带符号成交量 (buy=+vol, sell=-vol)
           (pl.when(pl.col("side") == 0).then(pl.col("volume")).otherwise(-pl.col("volume"))).sum().alias("cvd"),
       ]).collect(engine="streaming"))
rec("="*70)
rec("4) TRANSACTION (每样本聚合, n=%d)" % len(txn))
tc = txn["n_tx"]
rec(f"   每样本成交数: p10={tc.quantile(0.1):.0f} p50={tc.quantile(0.5):.0f} p90={tc.quantile(0.9):.0f} p99={tc.quantile(0.99):.0f} max={tc.max()}")
rec(f"   CVD (累计买卖净量): p10={txn['cvd'].quantile(0.1):+.0f} p50={txn['cvd'].quantile(0.5):+.0f} p90={txn['cvd'].quantile(0.9):+.0f}")

# ============ 5) 跨表关联 (含新微观结构量) ============
log("cross-table merge")
per = (ord_.join(txn, on="sample_id", how="full", coalesce=True)
       .join(lab.select(["sample_id", "target"]), on="sample_id", how="inner"))
per = per.with_columns([
    ((pl.col("n_sell")) / pl.col("n_order").clip(lower_bound=1)).alias("sell_ratio"),
    (pl.col("n_tx_sell") / pl.col("n_tx").clip(lower_bound=1)).alias("tx_sell_ratio"),
])
per = per.join(path.select(["sample_id", "mid_range", "mid_std", "spread_mean", "imb1_last",
                            "micro_gap_last", "micro_gap_mean", "bar_vol", "bidv1_last", "askv1_last"]),
               on="sample_id", how="left")
rec("="*70)
rec("5) 跨表关联 (n=%d) — 双轨 spearman" % len(per))
for c in ["n_order", "n_tx", "vol_order", "vol_tx", "sell_ratio", "tx_sell_ratio", "cvd",
          "vol_order_mean", "vol_tx_mean", "mid_range", "mid_std", "spread_mean", "imb1_last",
          "micro_gap_last", "micro_gap_mean", "bar_vol", "bidv1_last", "askv1_last"]:
    r = per.select(pl.corr(pl.col(c), pl.col("target"), method="spearman").alias("r")).item()
    ra = per.select(pl.corr(pl.col(c), pl.col("target").abs(), method="spearman").alias("r")).item()
    rec(f"   {c:16s} vs target={r:+.4f}  vs |target|={ra:+.4f}")
# 活动度分桶
per = per.with_columns(pl.col("n_order").qcut(5, labels=["Q1_lo","Q2","Q3","Q4","Q5_hi"]).alias("act_bin"))
act_abs = per.group_by("act_bin").agg([pl.col("target").abs().mean().alias("abs"), pl.col("target").mean().alias("mean"), pl.col("target").count().alias("n")]).sort("act_bin")
rec("\n   活动度分桶 → |target|:")
for r in act_abs.iter_rows(named=True):
    rec(f"   {r['act_bin']}: n={r['n']:7d} |t|mean={r['abs']:.2e} tmean={r['mean']:+.2e}")
# micro_gap 分桶 → 方向 (先剔除 null)
gap_bin = (per.filter(pl.col("micro_gap_last").is_not_null())
          .with_columns(pl.col("micro_gap_last").qcut(5, labels=["G1_neg","G2","G3","G4","G5_pos"]).alias("gap_bin")))
gap_abs = gap_bin.group_by("gap_bin").agg([pl.col("target").mean().alias("tmean"), pl.col("target").count().alias("n")]).sort("gap_bin")
rec("\n   micro_gap_last 分桶 → target 方向:")
for r in gap_abs.iter_rows(named=True):
    rec(f"   {r['gap_bin']}: n={r['n']:7d} tmean={r['tmean']:+.2e}")

# ============ 6) TEST 对比 (浅层) ============
log("test side (sample)")
tmkt = pl.read_ipc(f"{RAW}/test/market.feather", columns=["ask_price_1", "bid_price_1", "transaction_avgprice"])
rec("="*70)
rec("6) TEST 对比 (浅层)")
rec(f"   market rows={len(tmkt):,} 0价格ask1={((tmkt['ask_price_1']==0).mean()*100):.2f}% avgpriceNaN={(tmkt['transaction_avgprice'].is_null().mean()*100):.1f}%")
tord = pl.read_ipc(f"{RAW}/test/order.feather", columns=["side", "order_action"])
rec(f"   order rows={len(tord):,} cancel%={(tord['order_action']==1).mean()*100:.1f}% sell%={(tord['side']==1).mean()*100:.1f}%")
ttxn = pl.read_ipc(f"{RAW}/test/transaction.feather", columns=["side"])
rec(f"   tx rows={len(ttxn):,} sell%={(ttxn['side']==1).mean()*100:.1f}%")

# ============ 保存 ============
with open(f"{OUT}/stats_v2.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
rec(f"\nDONE -> {OUT}/stats_v2.txt  (elapsed {time.time()-T0:.0f}s)")
