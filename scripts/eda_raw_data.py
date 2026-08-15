# -*- coding: utf-8 -*-
"""MSCapital 原始数据集 EDA (2026-08-15)
层次: 1) label 深度分布与月度演化  2) market 盘口形态  3) order/tx 事件流统计
      4) 跨表关联 (活动度/imbalance vs target)  5) train vs test 漂移对比
输出: output/eda/*.png + stats.txt (全量聚合为主, 明细采样)
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
# 月度演化
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
# 极端值
rec(f"   |t|>2*std: {(t.abs()>2*t.std()).mean()*100:.2f}%   |t|>4*std: {(t.abs()>4*t.std()).mean()*100:.2f}%")
rec(f"   std 月度漂移: min={mstat['std'].min():.2e} max={mstat['std'].max():.2e} ratio={mstat['std'].max()/mstat['std'].min():.2f}")
rec(f"   mean 月度漂移: min={mstat['mean'].min():+.2e} max={mstat['mean'].max():+.2e}")

# ============ 2) MARKET ============
log("loading market (full)")
mkt = pl.read_ipc(f"{RAW}/train/market.feather")
rec("="*70)
rec("2) MARKET (rows=%d, cols=%d)" % (len(mkt), mkt.width))
# mid / spread / depth imbalance (先算派生, 注意 0 价格哨兵)
mkt = mkt.with_columns([
    ((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2).alias("mid1"),
    (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("spread1"),
    ((pl.col("bid_volume_1") - pl.col("ask_volume_1")) / (pl.col("bid_volume_1") + pl.col("ask_volume_1") + 1e-9)).alias("imb1"),
    ((pl.col("bid_volume_2") - pl.col("ask_volume_2")) / (pl.col("bid_volume_2") + pl.col("ask_volume_2") + 1e-9)).alias("imb2"),
])
valid = mkt.filter(pl.col("ask_price_1") > 0)
rec(f"   0价格哨兵: ask1={(mkt['ask_price_1']==0).mean()*100:.2f}% bid1={(mkt['bid_price_1']==0).mean()*100:.2f}%")
rec(f"   spread1 (有效行): p50={valid['spread1'].quantile(0.5):.2e} p90={valid['spread1'].quantile(0.9):.2e} p99={valid['spread1'].quantile(0.99):.2e}")
rec(f"   imb1: p1={valid['imb1'].quantile(0.01):.3f} p50={valid['imb1'].quantile(0.5):.3f} p99={valid['imb1'].quantile(0.99):.3f}")
rec(f"   imb2: p1={valid['imb2'].quantile(0.01):.3f} p50={valid['imb2'].quantile(0.5):.3f} p99={valid['imb2'].quantile(0.99):.3f}")
rec(f"   avgprice NaN: {(mkt['transaction_avgprice'].is_null().mean()*100):.1f}%")
rec(f"   tx_count: p50={mkt['transaction_count'].quantile(0.5):.0f} p90={mkt['transaction_count'].quantile(0.9):.0f} p99={mkt['transaction_count'].quantile(0.99):.0f} max={mkt['transaction_count'].max()}")
# 每样本 mid 路径统计: 幅度/波动
log("market per-sample path stats")
path = (mkt.group_by("sample_id").agg([
    (pl.col("mid1").max() - pl.col("mid1").min()).alias("mid_range"),
    pl.col("mid1").std().alias("mid_std"),
    (pl.col("mid1").last() - pl.col("mid1").first()).alias("mid_drift"),
    pl.col("mid1").first().alias("mid_first"),
    (pl.col("imb1") * pl.col("transaction_volume")).sum().alias("vwap_imb"),
    pl.col("transaction_volume").sum().alias("bar_vol"),
]))
rec(f"   每样本 mid_range: p50={path['mid_range'].quantile(0.5):.2e} p90={path['mid_range'].quantile(0.9):.2e} p99={path['mid_range'].quantile(0.99):.2e}")
rec(f"   每样本 mid_std:   p50={path['mid_std'].quantile(0.5):.2e} p90={path['mid_std'].quantile(0.9):.2e}")
rec(f"   每样本 mid_drift: p10={path['mid_drift'].quantile(0.1):+.2e} p50={path['mid_drift'].quantile(0.5):+.2e} p90={path['mid_drift'].quantile(0.9):+.2e}")
rec(f"   每样本 bar_vol:   p50={path['bar_vol'].quantile(0.5):.0f} p90={path['bar_vol'].quantile(0.9):.0f} p99={path['bar_vol'].quantile(0.99):.0f}")
# 与 label 合并 (全量)
labp = lab.rename({"target": "target"})
merged = path.join(labp.select(["sample_id", "target"]), on="sample_id", how="inner")
for c in ["mid_range", "mid_std", "bar_vol"]:
    sp = merged.select(pl.corr(pl.col(c), pl.col("target"), method="spearman").alias("r")).item()
    sp_abs = merged.select(pl.corr(pl.col(c), pl.col("target").abs(), method="spearman").alias("r")).item()
    rec(f"   spearman({c}, target)={sp:+.4f}  |target|={sp_abs:+.4f}")

# ============ 3) ORDER / TX ============
log("loading order")
ord_ = pl.read_ipc(f"{RAW}/train/order.feather")
rec("="*70)
rec("3) ORDER (rows=%d)" % len(ord_))
rec(f"   side counts: " + " ".join(f"{s}={n/len(ord_)*100:.1f}%" for s, n in ord_.group_by("side").len().sort("side").iter_rows()))
rec(f"   action counts: " + " ".join(f"{a}={n/len(ord_)*100:.1f}%" for a, n in ord_.group_by("order_action").len().sort("order_action").iter_rows()))
rec(f"   price: p1={ord_['price'].quantile(0.01):.4f} p50={ord_['price'].quantile(0.5):.4f} p99={ord_['price'].quantile(0.99):.4f} min={ord_['price'].min():.4f} max={ord_['price'].max():.4f}")
rec(f"   volume: 100倍数={(ord_['volume']%100==0).mean()*100:.1f}% p50={ord_['volume'].quantile(0.5):.0f} p99={ord_['volume'].quantile(0.99):.0f} max={ord_['volume'].max()}")
# 每样本事件数
oc = ord_.group_by("sample_id").len().get_column("len")
rec(f"   每样本事件数: p10={oc.quantile(0.1):.0f} p50={oc.quantile(0.5):.0f} p90={oc.quantile(0.9):.0f} p99={oc.quantile(0.99):.0f} max={oc.max()}")
# 事件间隔 (采样 2000 样本)
log("order inter-event intervals (sampled)")
sub_sids = rng.choice(lab['sample_id'].unique().to_numpy(), 2000, replace=False)
osub = ord_.filter(pl.col("sample_id").is_in(sub_sids)).sort(["sample_id", "seconds_before_predict"])
gap = (osub.group_by("sample_id").agg(pl.col("seconds_before_predict").diff().drop_nulls().alias("g"))
       .explode("g").filter(pl.col("g") > 0))
g = gap["g"]
rec(f"   到达间隔(s): p10={g.quantile(0.1):.3f} p50={g.quantile(0.5):.3f} p90={g.quantile(0.9):.3f} p99={g.quantile(0.99):.3f}")
rec(f"   间隔<0.01s占比: {(g<0.01).mean()*100:.1f}% (同秒批量)")
# 时间密度: 60s 内分布
hist = ord_.group_by(pl.col("seconds_before_predict").cast(pl.Int32).alias("sec_bin")).len().sort("sec_bin")
dens = hist.with_columns((pl.col("len") / (hist["len"].sum())).alias("p"))
for lo, hi in [(0, 5), (5, 15), (15, 30), (30, 45), (45, 60)]:
    p = dens.filter((pl.col("sec_bin") >= lo) & (pl.col("sec_bin") < hi))["p"].sum()
    rec(f"   order 事件密度 [{lo:2d},{hi:2d})s: {p*100:.1f}%")
# side x action 组合
rec("   side×action: " + " ".join(
    f"({s},{a})={n/len(ord_)*100:.1f}%" for s, a, n in
    ord_.group_by(["side", "order_action"]).len().sort(["side", "order_action"]).iter_rows()))

log("loading transaction")
txn = pl.read_ipc(f"{RAW}/train/transaction.feather")
rec("="*70)
rec("4) TRANSACTION (rows=%d)" % len(txn))
rec(f"   side counts: " + " ".join(f"{s}={n/len(txn)*100:.1f}%" for s, n in txn.group_by("side").len().sort("side").iter_rows()))
rec(f"   price: p1={txn['price'].quantile(0.01):.4f} p50={txn['price'].quantile(0.5):.4f} p99={txn['price'].quantile(0.99):.4f}")
rec(f"   volume: 100倍数={(txn['volume']%100==0).mean()*100:.1f}% p50={txn['volume'].quantile(0.5):.0f} p99={txn['volume'].quantile(0.99):.0f}")
tc = txn.group_by("sample_id").len().get_column("len")
rec(f"   每样本成交数: p10={tc.quantile(0.1):.0f} p50={tc.quantile(0.5):.0f} p90={tc.quantile(0.9):.0f} p99={tc.quantile(0.99):.0f} max={tc.max()}")
tsub = txn.filter(pl.col("sample_id").is_in(sub_sids)).sort(["sample_id", "seconds_before_predict"])
tgap = (tsub.group_by("sample_id").agg(pl.col("seconds_before_predict").diff().drop_nulls().alias("g"))
        .explode("g").filter(pl.col("g") > 0))["g"]
rec(f"   成交间隔(s): p10={tgap.quantile(0.1):.3f} p50={tgap.quantile(0.5):.3f} p90={tgap.quantile(0.9):.3f} p99={tgap.quantile(0.99):.3f}")
thist = txn.group_by(pl.col("seconds_before_predict").cast(pl.Int32).alias("sec_bin")).len().sort("sec_bin")
tdens = thist.with_columns((pl.col("len") / thist["len"].sum()).alias("p"))
for lo, hi in [(0, 5), (5, 15), (15, 30), (30, 45), (45, 60)]:
    p = tdens.filter((pl.col("sec_bin") >= lo) & (pl.col("sec_bin") < hi))["p"].sum()
    rec(f"   tx 事件密度 [{lo:2d},{hi:2d})s: {p*100:.1f}%")

# ============ 5) 跨表关联 ============
log("cross-table aggregation")
per = (ord_.group_by("sample_id").agg([
    pl.len().alias("n_order"),
    (pl.col("side") == 1).sum().alias("n_sell"),
    (pl.col("order_action") == 1).sum().alias("n_cancel"),
    pl.col("volume").sum().alias("vol_order"),
    (pl.col("side") == 1).sum().cast(pl.Float64).truediv(pl.len()).alias("sell_ratio"),
    ((pl.col("side") == 0) & (pl.col("order_action") == 0)).sum().alias("buy_add"),
    ((pl.col("side") == 1) & (pl.col("order_action") == 0)).sum().alias("sell_add"),
    ((pl.col("side") == 0) & (pl.col("order_action") == 1)).sum().alias("buy_cancel"),
    ((pl.col("side") == 1) & (pl.col("order_action") == 1)).sum().alias("sell_cancel"),
]).join(
    txn.group_by("sample_id").agg([
        pl.len().alias("n_tx"),
        (pl.col("side") == 1).sum().cast(pl.Float64).truediv(pl.len()).alias("tx_sell_ratio"),
        pl.col("volume").sum().alias("vol_tx"),
    ]), on="sample_id", how="outer").join(
    lab.select(["sample_id", "target"]), on="sample_id", how="inner"))
per = per.with_columns([
    ((pl.col("buy_add") - pl.col("sell_add")) / (pl.col("buy_add") + pl.col("sell_add") + 1e-9)).alias("add_imb"),
    ((pl.col("sell_cancel") - pl.col("buy_cancel")) / (pl.col("sell_cancel") + pl.col("buy_cancel") + 1e-9)).alias("cancel_imb"),
])
rec("="*70)
rec("5) 跨表关联 (n=%d)" % len(per))
for c in ["n_order", "n_tx", "vol_order", "vol_tx", "sell_ratio", "tx_sell_ratio", "add_imb", "cancel_imb"]:
    r = per.select(pl.corr(pl.col(c), pl.col("target"), method="spearman").alias("r")).item()
    ra = per.select(pl.corr(pl.col(c), pl.col("target").abs(), method="spearman").alias("r")).item()
    rec(f"   spearman({c}, target)={r:+.4f}  |target|={ra:+.4f}")
# 活动度分桶 → |target|
per = per.with_columns(pl.col("n_order").qcut(5, labels=["Q1_lo","Q2","Q3","Q4","Q5_hi"]).alias("act_bin"))
act_abs = per.group_by("act_bin").agg([pl.col("target").abs().mean().alias("abs"), pl.col("target").mean().alias("mean"), pl.col("target").count().alias("n")]).sort("act_bin")
rec("   活动度分桶 → |target|:")
for r in act_abs.iter_rows(named=True):
    rec(f"   {r['act_bin']}: n={r['n']:7d} |t|mean={r['abs']:.2e} tmean={r['mean']:+.2e}")

# ============ 6) TEST 对比 ============
log("test side (market sample)")
tmkt = pl.read_ipc(f"{RAW}/test/market.feather")
rec("="*70)
rec("6) TEST 对比")
rec(f"   market rows={len(tmkt):,} 0价格ask1={((tmkt['ask_price_1']==0).mean()*100):.2f}% avgpriceNaN={(tmkt['transaction_avgprice'].is_null().mean()*100):.1f}%")
tord = pl.read_ipc(f"{RAW}/test/order.feather")
rec(f"   order rows={len(tord):,} cancel%={(tord['order_action']==1).mean()*100:.1f}% sell%={(tord['side']==1).mean()*100:.1f}%")
ttxn = pl.read_ipc(f"{RAW}/test/transaction.feather")
rec(f"   tx rows={len(ttxn):,} sell%={(ttxn['side']==1).mean()*100:.1f}%")
# 价格水平对比
rec(f"   train order price p50={ord_['price'].quantile(0.5):.4f} vs test p50={tord['price'].quantile(0.5):.4f}")
rec(f"   train mkt ask1 p50={mkt['ask_price_1'].quantile(0.5):.4f} vs test p50={tmkt['ask_price_1'].quantile(0.5):.4f}")

# ============ 保存 ============
log("saving stats")
with open(f"{OUT}/stats.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
rec(f"\nDONE -> {OUT}/stats.txt")
