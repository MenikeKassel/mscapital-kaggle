# -*- coding: utf-8 -*-
"""数据集示例报告配图: 4 样本 × (600s mid 路径 + 60s 事件流时间线) (2026-08-15)
输出: docs/figures/sample_*.png
"""
import polars as pl
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RAW = r"D:\mscapital-forecasting\data\raw\train"
OUT = r"D:\mscapital-kaggle\docs\figures"
import os
os.makedirs(OUT, exist_ok=True)

SAMPLES = {
    "typical": 371716,
    "high_vol": 1209008,
    "low_act": 372208,
    "special_ask_empty": 372044,
}

for name, sid in SAMPLES.items():
    m = pl.read_ipc(f"{RAW}/market.feather").filter(pl.col("sample_id") == sid).sort("seconds_before_predict", descending=True)
    o = pl.read_ipc(f"{RAW}/order.feather").filter(pl.col("sample_id") == sid).sort("seconds_before_predict", descending=True)
    t = pl.read_ipc(f"{RAW}/transaction.feather").filter(pl.col("sample_id") == sid).sort("seconds_before_predict", descending=True)

    # mid 路径 (ask1>0 & bid1>0 才有效)
    mv = m.filter((pl.col("ask_price_1") > 0) & (pl.col("bid_price_1") > 0))
    mid = (mv["ask_price_1"] + mv["bid_price_1"]).to_numpy() / 2
    secs = mv["seconds_before_predict"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1.6, 1]})
    fig.suptitle(f"sample {sid} — {'typical' if name=='typical' else name}", fontsize=13, fontweight="bold")

    # 上: 600s mid 路径
    ax1.plot(secs, mid, "-", color="#1f77b4", lw=1.0, label="mid (600s)")
    ax1.set_xlabel("seconds_before_predict (600 → 0)")
    ax1.set_ylabel("mid price")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")
    ax1.set_xlim(600, 0)

    # 下: 60s 事件流 (order add/cancel + tx)
    os_ = o.filter(pl.col("seconds_before_predict") <= 60)
    ts_ = t.filter(pl.col("seconds_before_predict") <= 60)
    # order: add = 圆点, cancel = 叉
    for act, marker, lab in [(0, "o", "order add"), (1, "x", "order cancel")]:
        sub = os_.filter(pl.col("order_action") == act)
        if len(sub) == 0:
            continue
        col = np.where(sub["side"].to_numpy() == 0, "#2ca02c", "#d62728")
        ax2.scatter(sub["seconds_before_predict"].to_numpy(), sub["price"].to_numpy(),
                    c=col, marker=marker, s=18, alpha=0.75, label=lab)
    # tx: 方块
    if len(ts_) > 0:
        col = np.where(ts_["side"].to_numpy() == 0, "#2ca02c", "#d62728")
        ax2.scatter(ts_["seconds_before_predict"].to_numpy(), ts_["price"].to_numpy(),
                    c=col, marker="s", s=26, alpha=0.6, edgecolors="none", label="transaction")
    ax2.set_xlabel("seconds_before_predict (60 → 0)")
    ax2.set_ylabel("event price")
    ax2.set_xlim(60, 0)
    ax2.grid(alpha=0.3)

    # 合并图例: 颜色=方向 (buy/sell), 形状=事件类型 (add/cancel/tx)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markersize=8, label="buy (side=0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=8, label="sell (side=1)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888", markersize=8, label="○ order add"),
        Line2D([0], [0], marker="x", color="w", markerfacecolor="#888888", markeredgecolor="#888888", markersize=8, label="× order cancel"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#888888", markersize=8, label="■ transaction"),
    ]
    ax2.legend(handles=handles, loc="lower left", fontsize=8, ncol=1)

    # 标注特殊样本
    if name == "special_ask_empty":
        ax1.text(0.02, 0.95, "ask side empty (ask1=0) → mid=(0+bid)/2=0.5", transform=ax1.transAxes,
                 fontsize=10, color="darkred", verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="#fff3cd", alpha=0.9))
    if name == "high_vol":
        ax1.text(0.02, 0.95, "storm: mid range 10.6%, 881 orders / 299 tx", transform=ax1.transAxes,
                 fontsize=10, color="darkred", verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="#fdecea", alpha=0.9))

    plt.tight_layout()
    fp = f"{OUT}/sample_{name}.png"
    plt.savefig(fp, dpi=110)
    plt.close()
    print("saved", fp)
print("ALL DONE")
