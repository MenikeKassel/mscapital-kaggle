# 📈 MSCapital: Quantitative Market Microstructure EDA

## 🎯 Overview & Objectives
This notebook provides a comprehensive **Exploratory Data Analysis (EDA)** for the **MSCapital – Real Financial Market Forecasting** competition.

The objective is to predict future asset returns (`target`) using high-frequency market microstructure data. The dataset includes:
- **`label.feather`**: Sample targets and temporal mapping (`month` index).
- **`market.feather`**: Order book snapshots (Level 1 & Level 2) and time bars.
- **`order.feather`**: High-frequency raw order flow (new orders vs. cancellations).
- **`transaction.feather`**: Raw trade execution flow (aggressive buys vs. aggressive sells).

Given the data size (gigabytes and tens of millions of rows), we leverage **Polars** and **PyArrow** for memory-efficient loading.
## 🛠️ 1. Setup & Environment Configurations
## 📊 2. Target Variable & Temporal Distribution (`label.feather`)

We examine the target return distribution and check its stability across training months (`month` 0 to 70).
## 🏛️ 3. Order Book Microstructure (`market.feather`)

Order book snapshots provide insights into liquidity depth, bid-ask spread, and order imbalance:
- **Mid Price**: $\text{Mid Price} = \frac{\text{Ask}_1 + \text{Bid}_1}{2}$
- **Spread L1**: $\text{Spread}_1 = \text{Ask}_1 - \text{Bid}_1$
- **Order Imbalance**: $\text{Imbalance}_1 = \frac{\text{Bid Vol}_1 - \text{Ask Vol}_1}{\text{Bid Vol}_1 + \text{Ask Vol}_1 + \epsilon}$
## ⚡ 4. Raw Order Flow Dynamics (`order.feather`)

Raw order flow tracks individual orders within the last ~60 seconds before prediction:
- **`order_action`**: 0 = New Order, 1 = Cancel Order
- **`side`**: 0 = Buy, 1 = Sell
## 💥 5. Execution Flow & VWAP (`transaction.feather`)

Transaction data captures executed aggressive market orders. We compute the Volume Weighted Average Price (VWAP):
$$\text{VWAP} = \frac{\sum (\text{Price}_i \times \text{Volume}_i)}{\sum \text{Volume}_i}$$
## 📊 Additional Market Dynamics & Microstructure Visualizations

Beyond standard Order Book Level 1 snapshots and raw trade scatter plots, analyzing order book microstructure dynamics provides critical signals for short-term price movement prediction.

---

### 1. Cumulative Volume Delta (CVD)

**Definition:**
Cumulative Volume Delta measures the net pressure between aggressive buyers (market buy orders hitting the ask) and aggressive sellers (market sell orders hitting the bid) across time:

$$CVD_t = \sum_{i=1}^{t} \text{Signed Volume}_i, \quad \text{where } \text{Signed Volume} = \begin{cases} +V_i, & \text{if Buy } (\text{side} = 0) \\ -V_i, & \text{if Sell } (\text{side} = 1) \end{cases}$$

* **Visual Insight:** A rising CVD indicates buyer dominance, while a declining CVD signals sustained selling pressure.
* **Feature Potential:** Slope of CVD over short rolling windows ($5s, 10s, 30s$) and divergence between CVD and Mid-Price.
### 2. Micro-Price vs. Mid-Price Deviation

**Definition:**
While Mid-Price simply averages the top bid and ask prices, **Micro-Price** weights the prices by the available liquidity at Level 1:

$$\text{Micro Price} = \frac{\text{Bid Volume}_1 \cdot \text{Ask Price}_1 + \text{Ask Volume}_1 \cdot \text{Bid Price}_1}{\text{Bid Volume}_1 + \text{Ask Volume}_1}$$

* **Visual Insight:** When large order blocks sit on the Bid side, the Micro-Price shifts toward the Ask Price, capturing short-term order imbalance and impending price adjustments.
* **Feature Potential:** Real-time spread $(\text{Micro Price} - \text{Mid Price})$ serves as a high-frequency directional momentum feature.
### 3. Price-Level Volume Profile (Market Footprint)

**Definition:**
The Volume Profile aggregates total executed buy and sell volume at specific discrete price ticks throughout the sample window.

* **Visual Insight:** Identifies **High Volume Nodes (HVN)** where heavy liquidity exchange occurred (acting as dynamic support/resistance) versus **Low Volume Nodes (LVN)** where prices moved rapidly with minimal resistance.
* **Feature Potential:** Ratio of trade volume executed above vs. below the sample VWAP.
### 4. Top-of-Book Liquidity Depth Over Time

**Definition:**
Tracks the evolution of available share volume directly at `bid_volume_1` and `ask_volume_1` leading up to the prediction point ($t = 0$).

* **Visual Insight:** Rapid depletion of liquidity on one side of the book often signals institutional sweeps or order cancellations immediately preceding a price breakout.
* **Feature Potential:** Standard deviation, decay rate, and liquidity replenishment velocity at L1.
## 💡 Key Takeaways & Feature Engineering Roadmap

### 🔑 Key Takeaways:
1. **Target Dynamics**: Returns are zero-centered with volatility shifts across months (`month` 0–70).
2. **Microstructure Signals**: Level 1/2 **Order Book Imbalance (OBI)** and **Spread Compression** are strongly predictive.
3. **Flow Signals**: The ratio of **Order Cancellations to New Orders** and aggressive order imbalance in the last 60 seconds indicate short-term pressure.

### 🚀 Modeling Strategy:
- **Validation**: Use **GroupKFold** or **Purged GroupTimeSeries** split based on `month` to prevent temporal leakage.
- **Features**: Calculate rolling window statistics (`mean`, `std`, `skew`, `min`, `max`, `last`) over 5s, 10s, 30s, and 60s windows using Polars.