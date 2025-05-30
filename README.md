# 🍽️ Customer Segmentation & Promotion + Seasonal Trend Forecasting

This project applies **KMeans Clustering** for customer segmentation and **ARIMA Forecasting** for identifying seasonal trends in dish orders. It supports data-driven marketing and inventory strategies for the food service industry.

---

## 📊 Part 1: Customer Segmentation and Promotion (KMeans Clustering)

### 1️⃣ Data Loading and Preprocessing

- Loaded customer data from `sample_customer_data.csv`.
- Encoded categorical columns: `FavDish`, `TimePref`, and `Location` using `LabelEncoder`.
- Selected relevant features:
  - `TotalSpend`
  - `Frequency`
  - `AOV`
  - Encoded categorical fields

---

### 2️⃣ Feature Scaling

- Applied `StandardScaler` to normalize the features.
- Ensures all features contribute equally to clustering.

---

### 3️⃣ KMeans Clustering

- Used **KMeans algorithm**:
  - `n_clusters=4`
  - `random_state=42`
- Segmented customers based on behavior and preferences.
- Assigned a cluster label to each customer.

---

### 4️⃣ Cluster Profiling

Analyzed each cluster to determine:
- Average spend
- Visit frequency
- Average order value
- Common preferences

---

### 📌 Cluster Summary

| Cluster | Feature Summary                                                                 | Behavioral Insight                                                                 |
|---------|----------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| **0**   | Moderate to high spend, moderate frequency, moderate AOV                        | Steady spenders. Best for **loyalty programs** and "come back soon" offers.       |
| **1**   | Low spend, **high frequency**, low AOV                                           | Budget-conscious regulars. Upsell **combo deals** or value meals.                 |
| **2**   | **High spend**, low/moderate frequency, **high AOV**                            | Premium customers. Attracted by **lunch-time bundles** or occasion-based offers.  |
| **3**   | Low spend, low frequency, low AOV                                                | At-risk users. Use **reactivation coupons** and limited-time discounts.           |

---

### 5️⃣ Promotion Assignment

Mapped each cluster to a promotion:

| Cluster | Promotion Strategy                             |
|---------|------------------------------------------------|
| 0       | 🎁 Loyalty Points + "Come Back Soon" Offer     |
| 1       | 🍔 Upsell Combo Deals for Regulars             |
| 2       | 🕛 Lunch-Time Discount Bundle                  |
| 3       | 💸 Reactivation Coupon - Limited Time Offer    |

---

### 6️⃣ Visualization

- Created scatter plots for visualizing clusters and promotions.
- Saved results and visual assets for further analysis.

---

## 📈 Part 2: Seasonal Trend Identification (ARIMA Forecasting)

### 1️⃣ Data Loading and Preparation

- Loaded order data from `sample_orders.csv`.
- Converted `Date` column to datetime format.

---

### 2️⃣ Filtering and Aggregation

- Focused on a specific dish (e.g., **Pasta**).
- Aggregated **daily order quantities** for time series modeling.

---

### 3️⃣ ARIMA Model Fitting

- Applied **ARIMA model** with parameters:
  - `order=(2,1,2)`
- Captured trend, seasonality, and randomness in dish orders.

---

### 4️⃣ Forecasting

- Forecasted **next 14 days** of order quantities.
- Generated future date range and corresponding predictions.

---

### 5️⃣ Visualization & Seasonal Trend Analysis

- Plotted **actual vs. predicted** order quantities.
- Identified recurring **peaks, troughs, and patterns** in demand.

---

## ✅ Summary

- **KMeans Clustering** enables targeted promotions by segmenting customers based on behavior and preferences.
- **ARIMA Forecasting** reveals and predicts seasonal trends in dish orders, supporting smarter inventory and marketing planning.

---

## 📁 Files in this Project

```bash
📄 sample_customer_data.csv      # Customer feature data
📄 sample_orders.csv             # Dish order time series data
📈 cluster_profiles.csv          # Cluster-wise summary
🖼️  cluster_visuals.png         # Cluster scatter plots
📊 forecast_plot.png             # Forecast visualization
