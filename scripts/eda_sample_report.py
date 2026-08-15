# -*- coding: utf-8 -*-
"""数据集示例报告: 抽取代表性样本 v2 (2026-08-15)
四类样本: 典型 / 真高波动 / 超低活跃 / 特殊状态(ask全空=0.5价簇)
"""
import polars as pl
import json

RAW = r"D:\mscapital-forecasting\data\raw\train"
aud = pl.read_parquet(r"D:\mscapital-kaggle\output\p7amp_audit.parquet")
tgt = aud.select(["sample_id","month","y","p","mid_range","n_order","n_tx"])

# 1) 典型: n_order 中位附近
med = tgt["n_order"].median()
typ = tgt.filter((pl.col("n_order") > med*0.9) & (pl.col("n_order") < med*1.1)).sort("sample_id").head(1)
# 2) 真高波动: mid_range 非 null 最大
hi = tgt.filter(pl.col("mid_range").is_not_null()).sort("mid_range", descending=True).head(1)
# 3) 超低活跃: n_order==1
lo = tgt.filter(pl.col("n_order") == 1).head(1)
# 4) 特殊状态: 372044 (ask 全空, bid=1.0 巨量)
sel = {"typical": int(typ["sample_id"][0]), "high_vol": int(hi["sample_id"][0]),
       "low_act": int(lo["sample_id"][0]), "special_ask_empty": 372044}
print(json.dumps(sel, indent=2))

out = {"samples": sel, "data": {}}
for name, sid in sel.items():
    m = pl.read_ipc(f"{RAW}/market.feather").filter(pl.col("sample_id")==sid).sort("seconds_before_predict", descending=True)
    o = pl.read_ipc(f"{RAW}/order.feather").filter(pl.col("sample_id")==sid).sort("seconds_before_predict", descending=True)
    t = pl.read_ipc(f"{RAW}/transaction.feather").filter(pl.col("sample_id")==sid).sort("seconds_before_predict", descending=True)
    lab = tgt.filter(pl.col("sample_id")==sid).to_dicts()[0]
    out["data"][name] = {
        "label": lab,
        "market_n": len(m), "order_n": len(o), "tx_n": len(t),
        "market_head": m.head(10).to_dicts(),
        "market_tail": m.tail(2).to_dicts(),
        "order_head": o.head(20).to_dicts(),
        "tx_head": t.head(15).to_dicts(),
    }
with open(r"D:\mscapital-kaggle\output\eda\sample_report_data.json", "w") as f:
    json.dump(out, f, indent=1, default=str)
print("saved:", {k: (v["market_n"], v["order_n"], v["tx_n"]) for k, v in out["data"].items()})
