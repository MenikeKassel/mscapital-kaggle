# -*- coding: utf-8 -*-
"""GPT P7-AMP Step 2 实测: baseline amplitude audit (2026-08-15)
回答: baseline 预测是否已隐式吸收 heteroskedasticity?
  - corr(|p|, |y|)  baseline 幅度 vs 真实幅度
  - corr(mid_range, |p|)  mid_range 是否已反映进预测幅度
  - corr(mid_range, |y|)  复现 EDA +0.377
若 mid_range→|y| 强但 →|p| 弱 => 遗漏证据强, P7-AMP 有空间
"""
import polars as pl
import numpy as np, time
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)

RAW = r"D:\mscapital-forecasting\data\raw\train"
d = np.load(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
sids = d["sample_id"]; months = d["month"]; y = d["target"]; p = d["baseline_oof"]
log(f"canonical OOF: {len(sids)} rows, months {months.min()}-{months.max()}")
sid_arr = pl.Series("sample_id", sids)

# --- market: mid_range / mid_std / depth1 (仅合法 book, 过滤 0 哨兵) ---
log("scan market (filtered)")
mkt = (pl.scan_ipc(f"{RAW}/market.feather")
       .select(["sample_id","ask_price_1","bid_price_1","ask_volume_1","bid_volume_1"])
       .filter(pl.col("sample_id").is_in(sid_arr))
       .with_columns(((pl.col("ask_price_1")+pl.col("bid_price_1"))/2).alias("mid1"))
       .filter((pl.col("ask_price_1")>0)&(pl.col("bid_price_1")>0))
       .group_by("sample_id").agg([
           (pl.col("mid1").max()-pl.col("mid1").min()).alias("mid_range"),
           pl.col("mid1").std().alias("mid_std"),
           (pl.col("ask_volume_1")+pl.col("bid_volume_1")).mean().alias("depth1"),
       ]).collect())
log(f"market agg: {len(mkt)} rows")

# --- order / tx: n_order, n_tx ---
ord_ = (pl.scan_ipc(f"{RAW}/order.feather").select(["sample_id"])
        .filter(pl.col("sample_id").is_in(sid_arr))
        .group_by("sample_id").len().collect())
log(f"order agg: {len(ord_)} rows")
txn = (pl.scan_ipc(f"{RAW}/transaction.feather").select(["sample_id"])
       .filter(pl.col("sample_id").is_in(sid_arr))
       .group_by("sample_id").len().collect())
log(f"tx agg: {len(txn)} rows")

# --- merge ---
df = pl.DataFrame({"sample_id": sids, "month": months, "y": y, "p": p})
df = df.join(mkt, on="sample_id", how="left").join(ord_, on="sample_id", how="left").join(txn, on="sample_id", how="left")
df = df.rename({"len": "n_order"})
df = df.with_columns(pl.col("n_order").fill_null(0).alias("n_order"))
df = df.with_columns(pl.col("len_right").fill_null(0).alias("n_tx")) if "len_right" in df.columns else df
# polars join 重名处理: txn len 会被改名, 显式处理
cols = df.columns
if "len" in cols:
    df = df.rename({"len": "n_tx"})
log(f"merged: {len(df)} rows, cols={df.columns}")

# --- audit ---
log("audit corrs")
def sp(a, b):
    return df.select(pl.corr(pl.col(a), pl.col(b), method="spearman")).item()
print("="*70)
print("1) baseline 幅度吸收审计 (GPT Step 2)")
print(f"   corr(|p|, |y|)        = {sp('p','y') if False else df.select(pl.corr(pl.col('p').abs(), pl.col('y').abs(), method='spearman')).item():+.4f}")
print(f"   corr(mid_range, |y|)  = {sp('mid_range','y') if False else df.select(pl.corr(pl.col('mid_range'), pl.col('y').abs(), method='spearman')).item():+.4f}   (EDA 基准 +0.377)")
print(f"   corr(mid_range, |p|)  = {df.select(pl.corr(pl.col('mid_range'), pl.col('p').abs(), method='spearman')).item():+.4f}")
print(f"   corr(mid_std, |y|)    = {df.select(pl.corr(pl.col('mid_std'), pl.col('y').abs(), method='spearman')).item():+.4f}")
print(f"   corr(mid_std, |p|)    = {df.select(pl.corr(pl.col('mid_std'), pl.col('p').abs(), method='spearman')).item():+.4f}")
print(f"   corr(n_order, |y|)    = {df.select(pl.corr(pl.col('n_order'), pl.col('y').abs(), method='spearman')).item():+.4f}")
print(f"   corr(n_order, |p|)    = {df.select(pl.corr(pl.col('n_order'), pl.col('p').abs(), method='spearman')).item():+.4f}")
print(f"   corr(n_tx, |y|)       = {df.select(pl.corr(pl.col('n_tx'), pl.col('y').abs(), method='spearman')).item():+.4f}")
print(f"   corr(n_tx, |p|)       = {df.select(pl.corr(pl.col('n_tx'), pl.col('p').abs(), method='spearman')).item():+.4f}")
print(f"   corr(depth1, |y|)     = {df.select(pl.corr(pl.col('depth1'), pl.col('y').abs(), method='spearman')).item():+.4f}")
print(f"   corr(depth1, |p|)     = {df.select(pl.corr(pl.col('depth1'), pl.col('p').abs(), method='spearman')).item():+.4f}")
print()
print("2) 方向侧对照 (预测与特征的线性方向)")
print(f"   corr(p, y)            = {df.select(pl.corr(pl.col('p'), pl.col('y'), method='spearman')).item():+.4f}")
print(f"   corr(mid_range, y)    = {sp('mid_range','y'):+.4f}")
print()
print("3) 分月稳定性: corr(|p|,|y|) 与 corr(mid_range,|p|)")
mon = df.group_by("month").agg([
    pl.corr(pl.col("p").abs(), pl.col("y").abs(), method="spearman").alias("r_abs"),
    pl.corr(pl.col("mid_range"), pl.col("p").abs(), method="spearman").alias("r_mr_p"),
    pl.corr(pl.col("mid_range"), pl.col("y").abs(), method="spearman").alias("r_mr_y"),
]).sort("month")
for r in mon.iter_rows(named=True):
    if r["month"] % 5 == 0:
        print(f"   m{r['month']:3d}: corr(|p|,|y|)={r['r_abs']:+.4f}  corr(mr,|p|)={r['r_mr_p']:+.4f}  corr(mr,|y|)={r['r_mr_y']:+.4f}")
print()
print("4) 机制判定")
r_abs = df.select(pl.corr(pl.col("p").abs(), pl.col("y").abs(), method="spearman")).item()
r_mr_p = df.select(pl.corr(pl.col("mid_range"), pl.col("p").abs(), method="spearman")).item()
r_mr_y = df.select(pl.corr(pl.col("mid_range"), pl.col("y").abs(), method="spearman")).item()
if r_mr_p > 0.25:
    print(f"   mid_range→|p| 强 ({r_mr_p:+.3f}) => baseline 已部分吸收 volatility → P7-AMP 空间中等")
elif r_mr_p < 0.10:
    print(f"   mid_range→|p| 弱 ({r_mr_p:+.3f}) vs →|y| 强 ({r_mr_y:+.3f}) => 强遗漏证据 → P7-AMP 有真实空间")
else:
    print(f"   mid_range→|p| 中等 ({r_mr_p:+.3f})")
print(f"   corr(|p|,|y|)={r_abs:+.3f} (1.0=完美幅度校准, 0=无)")
# 保存
df.select(["sample_id","month","y","p","mid_range","mid_std","n_order","n_tx","depth1"]).write_parquet(r"D:\mscapital-kaggle\output\p7amp_audit.parquet")
print("saved -> output/p7amp_audit.parquet")
