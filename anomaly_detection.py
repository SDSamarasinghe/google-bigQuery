import pandas as pd
import numpy as np
from datetime import datetime

# Load the transaction data
# Adjust the file path if needed
DATA_PATH = 'sample_transaction_data.csv'

def detect_anomalies(data, window=7, threshold=2):
    anomalies = []
    # Group by DishName and sort by Date for rolling calculations
    for dish, group in data.groupby('DishName'):
        group = group.sort_values('Date')
        group['rolling_mean'] = group['Quantity'].rolling(window=window, min_periods=1).mean()
        group['rolling_std'] = group['Quantity'].rolling(window=window, min_periods=1).std(ddof=0)
        group['z_score'] = (group['Quantity'] - group['rolling_mean']) / group['rolling_std'].replace(0, np.nan)
        group['is_anomaly'] = group['z_score'].abs() > threshold
        anomalies.append(group)
    return pd.concat(anomalies)

def main():
    # Read CSV
    df = pd.read_csv(DATA_PATH)
    # Convert Date to datetime if not already
    if not np.issubdtype(df['Date'].dtype, np.datetime64):
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except:
            # If Date is epoch seconds
            df['Date'] = pd.to_datetime(df['Date'], unit='s')
    # Aggregate by DishName and Date
    agg = df.groupby(['DishName', 'Date'])['Quantity'].sum().reset_index()
    # Detect anomalies
    result = detect_anomalies(agg)
    # Show only anomalies
    anomalies = result[result['is_anomaly']]
    print('Detected anomalies:')
    print(anomalies[['DishName', 'Date', 'Quantity', 'rolling_mean', 'rolling_std', 'z_score', 'is_anomaly']])
    # Optionally save to CSV
    anomalies.to_csv('anomalies_detected.csv', index=False)

if __name__ == '__main__':
    main()
