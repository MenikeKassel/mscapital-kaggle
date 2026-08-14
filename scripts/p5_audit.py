# -*- coding: utf-8 -*-
"""P5 probe repo audit: data layout, feature availability, OOF anchors.

Prints everything the P5-A/B/C probes need to know, from the actual files.
"""
import numpy as np
import polars as pl

RAW = r"D:\mscapital-forecasting\data\raw\train"
PROC = r"D:\mscapital-forecasting\data\processed"
OUT = r"D:\mscapital-kaggle\output"

# ---- label ----
lab = pl.read_ipc(rf"{RAW}\label.feather")
print("label:", lab.shape, lab.columns)
print("  month range:", lab["month"].min(), "-", lab["month"].max())
cnt = lab.group_by("month").len().sort("month")
print("  months 0-20 rows:", cnt.filter(pl.col("month") <= 20)["len"].sum())
print("  months 21-70 rows:", cnt.filter((pl.col("month") >= 21) & (pl.col("month") <= 70))["len"].sum())
print("  months 71-108 rows:", cnt.filter(pl.col("month") >= 71)["len"].sum())
print("  target stats: mean=%.5f std=%.5f p50=%.5f p99=%.5f" % tuple(
    lab.select(pl.col("target").mean().alias("m"), pl.col("target").std().alias("s"),
               pl.col("target").quantile(0.5).alias("q50"), pl.col("target").quantile(0.99).alias("q99")).row(0)))

# ---- market feather ----
mk = pl.scan_ipc(rf"{RAW}\market.feather")
mk_cols = mk.collect_schema().names()
mk_rows = mk.select(pl.len()).collect().item()
print("\nmarket.feather:", mk_rows, "rows", mk_cols)
mk_lab = mk.select(pl.col("sample_id").unique().count()).collect().item()
print("  distinct sample_id:", mk_lab)

# ---- order / transaction ----
for f in ("order.feather", "transaction.feather"):
    p = rf"{RAW}\{f}"
    try:
        s = pl.scan_ipc(p)
        print(f"\n{f}: {s.select(pl.len()).collect().item()} rows, cols:", s.collect_schema().names())
        print("  distinct sample_id:", s.select(pl.col("sample_id").unique().count()).collect().item())
        secs = s.select(pl.col("seconds_before_predict").min().alias("mn"),
                        pl.col("seconds_before_predict").max().alias("mx")).collect().row(0)
        print("  seconds_before_predict range:", secs)
    except Exception as e:
        print(f"\n{f}: ERROR {e}")

# ---- f0726 152 features ----
fe = pl.read_parquet(rf"{PROC}\f0726_train_f32.parquet").sort("sample_id")
print("\nf0726_train_f32:", fe.shape)
print("  cols sample:", fe.columns[:8], "... total", len(fe.columns))
print("  sample_id range:", fe["sample_id"].min(), fe["sample_id"].max())
print("  NaN cols (worst 5):", fe.null_count().to_series().sort(descending=True).head(5).to_list())

# ---- market state (m05 / ReVol-lite context) ----
ms = pl.read_parquet(rf"{OUT}\m05_state\market_state_train.parquet")
print("\nmarket_state_train:", ms.shape, ms.columns[:12])
print("  sample_id range:", ms["sample_id"].min(), ms["sample_id"].max())
ms_ids = set(ms["sample_id"].to_list())
print("  distinct:", len(ms_ids))

# ---- canonical OOF ----
c = np.load(rf"{OUT}\canonical_residual_oof\canonical_residual_oof.npz")
print("\ncanonical_residual_oof keys:", list(c.keys()))
print("  baseline_oof: shape", c["baseline_oof"].shape, "finite", np.isfinite(c["baseline_oof"]).all())
print("  month range:", c["month"].min(), "-", c["month"].max(), "| source_train_end range:", c["source_train_end"].min(), "-", c["source_train_end"].max())
print("  target: mean=%.5f std=%.5f" % (c["target"].mean(), c["target"].std()))
print("  sample_id unique:", len(np.unique(c["sample_id"])))
# overlap with f0726
fe_ids = set(fe["sample_id"].to_list())
canon_ids = set(c["sample_id"].tolist())
print("  canonical ids in f0726:", len(canon_ids & fe_ids), "/", len(canon_ids))
print("  canonical ids in market_state:", len(canon_ids & ms_ids), "/", len(canon_ids))
# overlap with label
lab_ids = set(lab["sample_id"].to_list())
print("  canonical ids in label:", len(canon_ids & lab_ids))

# ---- v7-like pseudo preds ----
for f in ("rlps_final\\realmlp_pseudo_pred.npz", "rlps_v12\\v5_table_pseudo_pred.npz"):
    d = np.load(rf"{OUT}\{f}")
    print(f"\n{f}: keys={list(d.keys())} shape={d['pred'].shape} finite={np.isfinite(d['pred']).all()}")
