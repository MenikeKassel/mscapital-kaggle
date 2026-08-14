# -*- coding: utf-8 -*-
"""MSCapital 四文件数据生成结构取证 (采样版, 快)"""
import polars as pl
import numpy as np

RAW = "D:/mscapital-forecasting/data/raw"
rng = np.random.default_rng(42)

def sec_stats(name, df, sec_col="seconds_before_predict"):
    s = df[sec_col]
    print(f"  {name}.{sec_col}: dtype={df.schema[sec_col]} min={s.min()} max={s.max()} "
          f"p50={s.median():.1f} p99={s.quantile(0.99):.1f}")

print("="*80)
print("1) LABEL")
lab = pl.read_ipc(f"{RAW}/train/label.feather")
print(lab.schema)
print(f"  rows={len(lab)} months=[{lab['month'].min()},{lab['month'].max()}] "
      f"n_month={lab['month'].n_unique()}")
t = lab["target"]
print(f"  target: mean={t.mean():.2e} std={t.std():.2e} min={t.min():.2e} max={t.max():.2e} "
      f"pos%={(t>0).mean()*100:.1f}")
# 每月样本数
mc = lab.group_by("month").len().sort("month")
print("  每月样本数: min=%d max=%d 前3月=%s 后3月=%s" % (
    mc["len"].min(), mc["len"].max(), mc["len"].head(3).to_list(), mc["len"].tail(3).to_list()))

print("="*80)
print("2) MARKET (train) - 采样分析")
# 采样 sample_id 子集, 观察 per-sample 结构
all_sids = lab["sample_id"].unique().to_numpy()
sub = rng.choice(all_sids, 3000, replace=False)
m = pl.read_ipc(f"{RAW}/train/market.feather").filter(pl.col("sample_id").is_in(sub))
print("  schema:", m.schema)
print(f"  采样行数: {len(m)} ({3000} 样本)")
rc = m.group_by("sample_id").len()
print(f"  每样本行数: min={rc['len'].min()} p50={rc['len'].median()} p99={rc['len'].quantile(0.99)} max={rc['len'].max()}")
secs = m["seconds_before_predict"]
print(f"  seconds: min={secs.min()} max={secs.max()} dtype={m.schema['seconds_before_predict']}")
for lo, hi in [(0,10),(10,30),(30,60),(60,120),(120,300),(300,600)]:
    n = m.filter((secs>=lo)&(secs<hi)).height
    print(f"    [{lo:3d},{hi:3d})s: {n} 行 ({n/len(m)*100:.1f}%)")
# 快照间隔
m2 = m.sort(["sample_id","seconds_before_predict"])
gap = m2.group_by("sample_id").agg(pl.col("seconds_before_predict").diff().drop_nulls().alias("g")).explode("g")
g = gap["g"].filter(gap["g"]>0)
print(f"  快照间隔(s): min={g.min()} p50={g.median():.2f} p90={g.quantile(0.9):.2f} p99={g.quantile(0.99):.2f} max={g.max()}")
# 价格/量字段
for c in ["ask_price_1","bid_price_1","ask_price_2","bid_price_2","transaction_avgprice"]:
    s = m[c]
    print(f"  {c}: dtype={m.schema[c]} NaN%={s.is_null().mean()*100:.1f} min={s.min():.4f} "
          f"p50={s.median():.4f} p99={s.quantile(0.99):.4f} max={s.max():.4f}")
for c in ["ask_volume_1","bid_volume_1","ask_volume_2","bid_volume_2","transaction_volume","transaction_count"]:
    s = m[c]
    print(f"  {c}: dtype={m.schema[c]} NaN%={s.is_null().mean()*100:.1f} min={s.min()} "
          f"p50={s.median():.0f} p99={s.quantile(0.99):.0f} max={s.max():.0f}")
# tick 结构: bid_price_1 相邻不同值间距
bp = m["bid_price_1"].drop_nulls().unique().sort()
d = bp.diff().drop_nulls()
print(f"  bid_price_1 唯一值 {len(bp)} 个; 相邻间距: min={d.min():.6f} p50={d.median():.6f} "
      f"p90={d.quantile(0.9):.6f} max={d.max():.6f}")
# 快照时间是否整数秒
secs_f = secs.to_numpy()
print(f"  seconds 整数占比: {np.all(secs_f==np.round(secs_f))}")

print("="*80)
print("3) ORDER (train) - 采样分析")
o = pl.read_ipc(f"{RAW}/train/order.feather").filter(pl.col("sample_id").is_in(sub))
print("  schema:", o.schema)
print(f"  采样行数: {len(o)}")
secs = o["seconds_before_predict"]
print(f"  seconds: min={secs.min()} max={secs.max()} 整数={np.all(secs.to_numpy()==np.round(secs.to_numpy()))}")
print(f"  side 分布: {o.group_by('side').len().sort('side').to_dicts()}")
print(f"  order_action 分布: {o.group_by('order_action').len().sort('order_action').to_dicts()}")
print(f"  price: min={o['price'].min():.4f} p50={o['price'].median():.4f} max={o['price'].max():.4f} "
      f"NaN%={o['price'].is_null().mean()*100:.2f}")
print(f"  volume: min={o['volume'].min()} p50={o['volume'].median():.0f} p99={o['volume'].quantile(0.99):.0f} "
      f"max={o['volume'].max()} NaN%={o['volume'].is_null().mean()*100:.2f}")
op = o["price"].drop_nulls().unique().sort()
od = op.diff().drop_nulls()
print(f"  order price 唯一值 {len(op)}; 相邻间距 p50={od.median():.6f}")
# 每样本事件数
orc = o.group_by("sample_id").len()
print(f"  每样本订单事件数: min={orc['len'].min()} p50={orc['len'].median():.0f} p99={orc['len'].quantile(0.99):.0f} max={orc['len'].max()}")

print("="*80)
print("4) TRANSACTION (train) - 采样分析")
tx = pl.read_ipc(f"{RAW}/train/transaction.feather").filter(pl.col("sample_id").is_in(sub))
print("  schema:", tx.schema)
print(f"  采样行数: {len(tx)}")
secs = tx["seconds_before_predict"]
print(f"  seconds: min={secs.min()} max={secs.max()} 整数={np.all(secs.to_numpy()==np.round(secs.to_numpy()))}")
print(f"  side 分布: {tx.group_by('side').len().sort('side').to_dicts()}")
print(f"  price: min={tx['price'].min():.4f} p50={tx['price'].median():.4f} max={tx['price'].max():.4f} "
      f"NaN%={tx['price'].is_null().mean()*100:.2f}")
print(f"  volume: min={tx['volume'].min()} p50={tx['volume'].median():.0f} p99={tx['volume'].quantile(0.99):.0f} "
      f"max={tx['volume'].max()} NaN%={tx['volume'].is_null().mean()*100:.2f}")
txc = tx.group_by("sample_id").len()
print(f"  每样本成交事件数: min={txc['len'].min()} p50={txc['len'].median():.0f} p99={txc['len'].quantile(0.99):.0f} max={txc['len'].max()}")
# 成交价 tick
tp = tx["price"].drop_nulls().unique().sort()
td = tp.diff().drop_nulls()
print(f"  tx price 唯一值 {len(tp)}; 相邻间距 p50={td.median():.6f}")

print("="*80)
print("5) 交叉关系")
# 同一个样本里: order/tx 的秒集合 vs market 秒集合 是否对齐
sid = sub[0]
mm = m.filter(pl.col("sample_id")==sid).sort("seconds_before_predict")
oo = o.filter(pl.col("sample_id")==sid).sort("seconds_before_predict")
tt = tx.filter(pl.col("sample_id")==sid).sort("seconds_before_predict")
print(f"  样本 {sid}: market {mm.height} 行 (秒 {mm['seconds_before_predict'].min()}..{mm['seconds_before_predict'].max()}), "
      f"order {oo.height} 行, tx {tt.height} 行")
print(f"  market 秒前10: {mm['seconds_before_predict'].head(10).to_list()}")
print(f"  order 秒前10: {oo['seconds_before_predict'].head(10).to_list()}")
print(f"  tx 秒前10: {tt['seconds_before_predict'].head(10).to_list()}")
# market 的 transaction_avgprice vs tx 表: 对齐吗?
if tt.height>0 and mm.height>0:
    mt = mm["transaction_avgprice"].drop_nulls()
    if len(mt)>0:
        print(f"  market.transaction_avgprice 非空 {len(mt)}/{mm.height}, 例: {[round(x,4) for x in mt.head(5).to_list()]}")
    print(f"  tx 表 price 例: {[round(x,4) for x in tt['price'].head(5).to_list()]}")
# 时间重叠: market 快照秒 与 order 事件秒 的交集比例
mset = set(mm["seconds_before_predict"].to_list()); oset = set(oo["seconds_before_predict"].to_list())
inter = mset & oset
print(f"  market∩order 秒交集: {len(inter)} / order {len(oset)} (={len(inter)/max(len(oset),1)*100:.1f}%)")

print("="*80)
print("7) 价格 0 哨兵 & spread 结构")
z = m.with_columns(
    (pl.col("ask_price_1")==0).alias("az1"),
    (pl.col("bid_price_1")==0).alias("bz1"),
    (pl.col("ask_price_2")==0).alias("az2"),
    (pl.col("bid_price_2")==0).alias("bz2"),
)
for c in ["az1","bz1","az2","bz2"]:
    zc = z.filter(pl.col(c))
    print(f"  {c}: {zc.height} 行 ({zc.height/len(z)*100:.2f}%)")
# price==0 行对应的 volume
z1 = z.filter(pl.col("az1"))
if z1.height>0:
    print(f"  ask_price_1==0 行的 ask_volume_1: min={z1['ask_volume_1'].min()} p50={z1['ask_volume_1'].median()} "
          f"max={z1['ask_volume_1'].max()}, ==0 占比 {(z1['ask_volume_1']==0).mean()*100:.1f}%")
# 有效报价行的 spread
v = z.filter((pl.col("ask_price_1")>0)&(pl.col("bid_price_1")>0)).with_columns(
    (pl.col("ask_price_1")-pl.col("bid_price_1")).alias("sp"))
sp = v["sp"]
print(f"  有效报价行 {v.height}/{len(z)}: spread min={sp.min():.2e} p50={sp.median():.2e} "
      f"p90={sp.quantile(0.9):.2e} p99={sp.quantile(0.99):.2e} max={sp.max():.2e}")
print(f"  spread==0 (盘口重叠) 占比: {(sp==0).mean()*100:.2f}%")
print(f"  spread<0 (交叉盘口) 占比: {(sp<0).mean()*100:.2f}%")

print("="*80)
print("8) 交易聚合一致性 (market bar vs tx 事件)")
# 用样本 738512: market.transaction_count 总和 vs tx 表行数
sid2 = 738512
mm3 = m.filter(pl.col("sample_id")==sid2)
tt3 = tx.filter(pl.col("sample_id")==sid2)
print(f"  样本 {sid2}: market.transaction_count 总和 = {mm3['transaction_count'].sum()}, tx 表行数 = {tt3.height}")
print(f"  market.transaction_volume 总和 = {mm3['transaction_volume'].sum()}, tx 表 volume 总和 = {tt3['volume'].sum()}")
# 60s 内 market bar 的 tx 聚合 vs 60s 内 tx 表
mm60 = mm3.filter(pl.col("seconds_before_predict")<=60)
tt60 = tt3.filter(pl.col("seconds_before_predict")<=60)
print(f"  ≤60s: market bar count 总和 = {mm60['transaction_count'].sum()} vs tx 表 {tt60.height} 行; "
      f"volume {mm60['transaction_volume'].sum()} vs {tt60['volume'].sum()}")

print("="*80)
print("9) order/tx 价格网格精确值")
for name, df in [("order", o), ("tx", tx)]:
    p = df["price"].drop_nulls().unique().sort()
    d = p.diff().drop_nulls().filter(pl.col("") if False else p.diff().drop_nulls()>0)
    import collections
    cnt = collections.Counter(d.to_list())
    top = cnt.most_common(5)
    print(f"  {name} price 相邻间距 top5 (值,次数): {[(round(k,8),v) for k,v in top]}")
    # 是否 1e-6 的整数倍
    pv = p.to_numpy()
    mult = np.abs(pv / 1e-6 - np.round(pv / 1e-6))
    print(f"  {name}: 价格是 1e-6 整数倍的比例: {(mult<1e-3).mean()*100:.2f}% (检查 {min(len(pv),20000)} 值)")
    mv = np.abs(pv / 1e-5 - np.round(pv / 1e-5))
    print(f"  {name}: 价格是 1e-5 整数倍的比例: {(mv<1e-3).mean()*100:.2f}%")

print("="*80)
print("10) 成交量网格")
for name, df in [("order", o), ("tx", tx)]:
    vol = df["volume"].to_numpy()
    m100 = (vol % 100 == 0).mean()*100
    m1 = (vol % 1 == 0).mean()*100
    print(f"  {name} volume: 100 整数倍占比 {m100:.1f}%, 全部整数 {m1:.0f}%")
    mc = collections.Counter(vol[:200000].tolist())
    print(f"    top5 常见量: {mc.most_common(5)}")

print("="*80)
print("6) 价格归一化检查 (跨样本)")
# 多个样本的 mid 价格中位数
sub2 = rng.choice(all_sids, 500, replace=False)
mm2 = pl.read_ipc(f"{RAW}/train/market.feather").filter(pl.col("sample_id").is_in(sub2))
mmid = mm2.with_columns(((pl.col("ask_price_1")+pl.col("bid_price_1"))/2).alias("mid")) \
         .group_by("sample_id").agg(pl.col("mid").median().alias("m"))
print(f"  500 样本 mid 中位数: min={mmid['m'].min():.4f} p50={mmid['m'].median():.4f} max={mmid['m'].max():.4f}")
print(f"  其中 <0.7 的样本数: {(mmid['m']<0.7).sum()} / {mmid.height}")
