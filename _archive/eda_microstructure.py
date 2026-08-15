import gc
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

warnings.filterwarnings("ignore")

# Visualization Styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.titlesize"] = 14
plt.rcParams["axes.titlesize"] = 12

DATA_DIR = "data"
print("Environment and libraries loaded successfully!")
import os

# Update DATA_DIR automatically if running on Kaggle
KAGGLE_PATH = "/kaggle/input/competitions/ms-capital-real-financial-market-forecasting"
if os.path.exists(KAGGLE_PATH):
    DATA_DIR = KAGGLE_PATH
else:
    DATA_DIR = "data"

print(f"Using DATA_DIR: {DATA_DIR}\n")

# Load labels (Main target table)
train_labels = pl.read_ipc(f"{DATA_DIR}/train/label.feather")

print(f"--- TRAIN LABELS ---")
print(f"Shape: {train_labels.shape}")
print(train_labels.head(3))
print("\n" + "="*50 + "\n")

# Lazy loading / schema inspection for heavy files
market_schema = pl.scan_ipc(f"{DATA_DIR}/train/market.feather").schema
order_schema = pl.scan_ipc(f"{DATA_DIR}/train/order.feather").schema
transaction_schema = pl.scan_ipc(f"{DATA_DIR}/train/transaction.feather").schema



print("--- MARKET FEATHER SCHEMA ---")
for col, dtype in market_schema.items():
    print(f"  {col}: {dtype}")

print("\n--- ORDER FEATHER SCHEMA ---")
for col, dtype in order_schema.items():
    print(f"  {col}: {dtype}")

print("\n--- TRANSACTION FEATHER SCHEMA ---")
for col, dtype in transaction_schema.items():
    print(f"  {col}: {dtype}")
# Load label metadata using Polars with memory_map=False
labels_df = pl.read_ipc(f"{DATA_DIR}/train/label.feather", memory_map=False)

print(f"Total Training Samples: {labels_df.height:,}")
print(f"Month Index Range: {labels_df['month'].min()} to {labels_df['month'].max()}")
labels_df.head()
# Convert to pandas for plotting
labels_pd = labels_df.to_pandas()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 1. Target Distribution
sns.histplot(
    labels_pd["target"],
    bins=120,
    kde=True,
    ax=axes[0],
    color="#1f77b4",
    edgecolor="none",
)
axes[0].set_title(
    f"Target Distribution (Mean: {labels_pd['target'].mean():.5f}, Std: {labels_pd['target'].std():.5f})"
)
axes[0].set_xlabel("Target (Future Return)")
axes[0].set_ylabel("Frequency")

# 2. Target Mean & Volatility by Month
month_stats = (
    labels_pd.groupby("month")["target"]
    .agg(["mean", "std"])
    .reset_index()
)
axes[1].plot(
    month_stats["month"],
    month_stats["mean"],
    label="Mean Return",
    color="#1f77b4",
    lw=1.5,
)
axes[1].fill_between(
    month_stats["month"],
    month_stats["mean"] - month_stats["std"],
    month_stats["mean"] + month_stats["std"],
    color="#1f77b4",
    alpha=0.15,
    label="±1 Std (Market Volatility)",
)
axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
axes[1].set_title("Target Stability Across Months (0 to 70)")
axes[1].set_xlabel("Month Index")
axes[1].set_ylabel("Target Return")
axes[1].legend()

plt.tight_layout()
plt.show()
# Select a sample_id
target_sample = labels_df["sample_id"].head(1)[0]

# Use scan_ipc (Lazy Execution) so Polars filters BEFORE loading data into RAM
market_sample = (
    pl.scan_ipc(f"{DATA_DIR}/train/market.feather")
    .filter(pl.col("sample_id") == target_sample)
    .collect()  # Execute the query only for the filtered subset
    .to_pandas()
)

# Feature engineering for visualization
market_sample["mid_price"] = (
    market_sample["ask_price_1"] + market_sample["bid_price_1"]
) / 2
market_sample["spread_1"] = (
    market_sample["ask_price_1"] - market_sample["bid_price_1"]
)
market_sample["book_imbalance_1"] = (
    market_sample["bid_volume_1"] - market_sample["ask_volume_1"]
) / (market_sample["bid_volume_1"] + market_sample["ask_volume_1"] + 1e-5)

# Sort time axis (closer to prediction = smaller seconds)
market_sample = market_sample.sort_values("seconds_before_predict", ascending=False)

# Visualization
fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

# 1. Level 1 Prices
axes[0].plot(
    market_sample["seconds_before_predict"],
    market_sample["ask_price_1"],
    label="Ask 1",
    color="#d62728",
    alpha=0.7,
)
axes[0].plot(
    market_sample["seconds_before_predict"],
    market_sample["bid_price_1"],
    label="Bid 1",
    color="#2ca02c",
    alpha=0.7,
)
axes[0].plot(
    market_sample["seconds_before_predict"],
    market_sample["mid_price"],
    label="Mid Price",
    color="black",
    linestyle="--",
)
axes[0].set_title(f"L1 Order Book Dynamics — Sample ID: {target_sample}")
axes[0].set_ylabel("Price")
axes[0].legend()

# 2. Bid-Ask Spread
axes[1].plot(
    market_sample["seconds_before_predict"],
    market_sample["spread_1"],
    color="#ff7f0e",
)
axes[1].set_title("Bid-Ask Spread Dynamics (L1)")
axes[1].set_ylabel("Spread")

# 3. Imbalance
axes[2].plot(
    market_sample["seconds_before_predict"],
    market_sample["book_imbalance_1"],
    color="#9467bd",
)
axes[2].axhline(0, color="gray", linestyle="--")
axes[2].set_title("L1 Order Book Imbalance [-1.0 to +1.0]")
axes[2].set_xlabel("Seconds Before Prediction")
axes[2].set_ylabel("Imbalance")

plt.tight_layout()
plt.show()
# Load order flow using Lazy execution (scan_ipc) to save RAM
order_sample = (
    pl.scan_ipc(f"{DATA_DIR}/train/order.feather")
    .filter(pl.col("sample_id") == target_sample)
    .collect()
    .to_pandas()
)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 1. Order Action Breakdown
action_counts = order_sample["order_action"].value_counts()
action_labels = {0: "New Order (0)", 1: "Cancel Order (1)"}
pie_labels = [action_labels.get(idx, str(idx)) for idx in action_counts.index]

axes[0].pie(
    action_counts,
    labels=pie_labels,
    autopct="%1.1f%%",
    colors=["#2ca02c", "#d62728"],
    startangle=140,
)
axes[0].set_title("Order Action Distribution")

# 2. Volume Breakdown by Side & Action
sns.barplot(
    data=order_sample,
    x="side",
    y="volume",
    hue="order_action",
    estimator=sum,
    ax=axes[1],
    palette="Set2",
)

# Fix for Matplotlib ValueError: set ticks explicitly before labels
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["Buy (0)", "Sell (1)"])
axes[1].set_title("Cumulative Volume by Side and Action")
axes[1].set_xlabel("Order Side")
axes[1].set_ylabel("Total Share Volume")

plt.tight_layout()
plt.show()
# 1. Lazy load with scan_ipc to avoid RAM crash and IPC warnings
tx_sample = (
    pl.scan_ipc(f"{DATA_DIR}/train/transaction.feather")
    .filter(pl.col("sample_id") == target_sample)
    .collect()
    .to_pandas()
)

# 2. Calculate VWAP
tx_sample["pv"] = tx_sample["price"] * tx_sample["volume"]
total_volume = tx_sample["volume"].sum()
vwap_price = tx_sample["pv"].sum() / total_volume if total_volume > 0 else np.nan

fig, ax = plt.subplots(figsize=(16, 5))

buy_tx = tx_sample[tx_sample["side"] == 0]
sell_tx = tx_sample[tx_sample["side"] == 1]

# 3. Plot Aggressive Buys
if not buy_tx.empty:
    buy_median = buy_tx["volume"].median()
    buy_sizes = (
        (buy_tx["volume"] / buy_median * 20).clip(10, 300)
        if buy_median > 0
        else 20
    )
    ax.scatter(
        buy_tx["seconds_before_predict"],
        buy_tx["price"],
        s=buy_sizes,
        color="#2ca02c",
        alpha=0.5,
        label="Aggressive Buy (0)",
    )

# 4. Plot Aggressive Sells
if not sell_tx.empty:
    sell_median = sell_tx["volume"].median()
    sell_sizes = (
        (sell_tx["volume"] / sell_median * 20).clip(10, 300)
        if sell_median > 0
        else 20
    )
    ax.scatter(
        sell_tx["seconds_before_predict"],
        sell_tx["price"],
        s=sell_sizes,
        color="#d62728",
        alpha=0.5,
        label="Aggressive Sell (1)",
    )

# 5. Plot VWAP line
if not np.isnan(vwap_price):
    ax.axhline(
        vwap_price,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        label=f"Sample VWAP ({vwap_price:.2f})",
    )

ax.set_title(f"Executed Transaction Flow — Sample ID: {target_sample}")
ax.set_xlabel("Seconds Before Prediction")
ax.set_ylabel("Execution Price")
ax.legend()

plt.tight_layout()
plt.show()
# Обчислюємо дельту об'єму для кожної угоди (+ для Buy, - для Sell)
tx_sample["signed_volume"] = np.where(tx_sample["side"] == 0, tx_sample["volume"], -tx_sample["volume"])

# Сортуємо за часом (від минулого до моменту предикту)
tx_sample_sorted = tx_sample.sort_values("seconds_before_predict", ascending=False).copy()
tx_sample_sorted["cvd"] = tx_sample_sorted["signed_volume"].cumsum()

fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(tx_sample_sorted["seconds_before_predict"], tx_sample_sorted["cvd"], color="#1f77b4", linewidth=2)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.fill_between(tx_sample_sorted["seconds_before_predict"], tx_sample_sorted["cvd"], 0, 
                 where=(tx_sample_sorted["cvd"] >= 0), color="#2ca02c", alpha=0.2)
ax.fill_between(tx_sample_sorted["seconds_before_predict"], tx_sample_sorted["cvd"], 0, 
                 where=(tx_sample_sorted["cvd"] < 0), color="#d62728", alpha=0.2)

ax.set_title(f"Cumulative Volume Delta (CVD) — Sample ID: {target_sample}")
ax.set_xlabel("Seconds Before Prediction")
ax.set_ylabel("Net Cumulative Volume")
plt.tight_layout()
plt.show()
market_sample["micro_price"] = (
    (market_sample["bid_volume_1"] * market_sample["ask_price_1"]) +
    (market_sample["ask_volume_1"] * market_sample["bid_price_1"])
) / (market_sample["bid_volume_1"] + market_sample["ask_volume_1"] + 1e-5)

fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(market_sample["seconds_before_predict"], market_sample["mid_price"], label="Mid Price", color="black", linestyle="--")
ax.plot(market_sample["seconds_before_predict"], market_sample["micro_price"], label="Micro Price (Volume Weighted)", color="#9467bd")

ax.set_title(f"Micro-price vs Mid-price Deviation — Sample ID: {target_sample}")
ax.set_xlabel("Seconds Before Prediction")
ax.set_ylabel("Price")
ax.legend()
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize=(8, 5))

# Агрегуємо об'єм за ціновими рівнями
vol_profile = tx_sample.groupby(["price", "side"])["volume"].sum().unstack(fill_value=0)

if 0 in vol_profile.columns:
    ax.barh(vol_profile.index, vol_profile[0], color="#2ca02c", alpha=0.6, label="Buy Volume", height=0.01)
if 1 in vol_profile.columns:
    ax.barh(vol_profile.index, -vol_profile[1], color="#d62728", alpha=0.6, label="Sell Volume", height=0.01)

ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
ax.set_title(f"Volume Profile by Price Level — Sample ID: {target_sample}")
ax.set_xlabel("Sell Volume  ←  0  →  Buy Volume")
ax.set_ylabel("Execution Price")
ax.legend()
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize=(16, 4))

ax.plot(market_sample["seconds_before_predict"], market_sample["bid_volume_1"], label="Bid Volume 1", color="#2ca02c")
ax.plot(market_sample["seconds_before_predict"], market_sample["ask_volume_1"], label="Ask Volume 1", color="#d62728")

ax.set_title(f"Top-of-Book Liquidity Depth — Sample ID: {target_sample}")
ax.set_xlabel("Seconds Before Prediction")
ax.set_ylabel("Available Volume")
ax.legend()
plt.tight_layout()
plt.show()