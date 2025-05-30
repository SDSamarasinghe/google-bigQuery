import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# ----------------------------
# Step 1: Load sample data from CSV
# ----------------------------

# Adjust the path if needed
csv_path = '/Users/macbook/arima_forcast/sample_orders.csv'
df = pd.read_csv(csv_path)

# Ensure correct date type
df['Date'] = pd.to_datetime(df['Date'])

# ----------------------------
# Step 2: Filter for one dish and aggregate daily order counts
# ----------------------------

dish_name = 'Pasta'  # change this to try others
df_dish = df[df['DishName'] == dish_name].copy()
df_dish = df_dish.groupby('Date')['Quantity'].sum().asfreq('D').fillna(0)

# ----------------------------
# Step 3: Fit ARIMA Model
# ----------------------------

model = ARIMA(df_dish, order=(2, 1, 2))  # (p,d,q) - can be tuned
model_fit = model.fit()

# ----------------------------
# Step 4: Forecast next 14 days
# ----------------------------

forecast = model_fit.forecast(steps=14)  # <-- Forecasts next 14 days' order quantities
forecast_dates = pd.date_range(df_dish.index[-1] + pd.Timedelta(days=1), periods=14)  # <-- Generates future dates

# ----------------------------
# Step 5: Plot the results
# ----------------------------

plt.figure(figsize=(10, 5))
plt.plot(df_dish.index, df_dish, label='Observed', marker='o')
plt.plot(forecast_dates, forecast, label='Forecast', color='orange', marker='x')
plt.title(f"ARIMA Forecast for {dish_name}")
plt.xlabel("Date")
plt.ylabel("Order Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
