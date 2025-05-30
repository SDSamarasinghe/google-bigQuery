from flask import Flask, jsonify, send_from_directory
import pandas as pd
import numpy as np
from anomaly_detection import detect_anomalies, DATA_PATH
import os

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    try:
        return app.send_static_file('anomaly_frontend.html')
    except Exception as e:
        print(f"Error serving anomaly_frontend.html: {e}")
        return f"Error: {e}", 500

@app.route('/health')
def health():
    return 'OK', 200

@app.route('/detect-anomalies')
def detect_anomalies_api():
    # Load and process data
    df = pd.read_csv(DATA_PATH)
    if not np.issubdtype(df['Date'].dtype, np.datetime64):
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except:
            df['Date'] = pd.to_datetime(df['Date'], unit='s')
    agg = df.groupby(['DishName', 'Date'])['Quantity'].sum().reset_index()
    result = detect_anomalies(agg)
    anomalies = result[result['is_anomaly']]
    # Format for frontend
    anomalies_json = [
        {
            'DishName': row.DishName,
            'Date': row.Date.isoformat() if hasattr(row.Date, 'isoformat') else str(row.Date),
            'Quantity': row.Quantity,
            'is_anomaly': bool(row.is_anomaly),
            'z_score': float(row.z_score) if not pd.isnull(row.z_score) else None
        }
        for row in anomalies.itertuples()
    ]
    return jsonify(anomalies_json)

import math

@app.route('/customer-segments')
def customer_segments():
    # Read KMeans results
    df = pd.read_csv('kmeans_results.csv')
    # Map cluster to action
    def describe(row):
        if row['AOV'] > 100:
            return (
                "VIP: High average order value (AOV > $100), but low frequency. "
                "These are big spenders who visit rarely. Invite to exclusive event or offer premium upsell."
            )
        elif row['Frequency'] > 5:
            return (
                "Loyal Regular: Visits often (frequency > 5), moderate spend per visit. "
                "These are your regulars. Offer loyalty rewards or cross-sell to increase basket size."
            )
        elif row['AOV'] < 30:
            return (
                "Budget Customer: Low average order value (AOV < $30), moderate frequency. "
                "They are price-sensitive. Promote affordable combos or value deals."
            )
        else:
            return (
                "New/Occasional: Low frequency, moderate spend. "
                "These are new or infrequent customers. Send welcome or re-engagement offers."
            )
    df['note'] = df.apply(describe, axis=1)
    # Optionally decode encoded fields if you have mapping
    result = df.to_dict(orient='records')
    # Convert any NaN to None for JSON serialization
    for r in result:
        for k, v in r.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                r[k] = None
    return jsonify(result)

from flask import request

@app.route('/forecast', methods=['GET'])
def forecast():
    # Get parameters from query string
    try:
        horizon = int(request.args.get('horizon', 14))
        confidence_level = float(request.args.get('confidence_level', 0.9))
        dish_name = request.args.get('dish_name', '').strip()
    except Exception:
        return {'error': 'Invalid input for horizon, confidence_level, or dish_name.'}, 400
    # Build the query (for info/demo)
    dish_filter = f"WHERE DishName = '{dish_name}'" if dish_name else ''
    query = (
        f"SELECT * FROM ML.FORECAST(\n"
        f"  MODEL `restaurant-forecast.restaurantforecast.arima_model`,\n"
        f"  STRUCT({horizon} AS horizon, {confidence_level} AS confidence_level)\n"
        f") {dish_filter}"
    )
    # Run query on BigQuery and get response
    # For demo, assume we have a function to run the query and get the response
    response = run_bigquery_query(query)
    # Parse rows and schema
    rows = response['rows']
    schema = response['schema']
    # Convert timestamps
    for row in rows:
        for i, field in enumerate(schema):
            if field['type'] == 'TIMESTAMP':
                row['f'][i]['v'] = row['f'][i]['v'].isoformat()
    # Format for frontend
    forecast_json = [
        {
            field['name']: row['f'][i]['v']
            for i, field in enumerate(schema)
        }
        for row in rows
    ]
    return {'query': query, 'forecast': forecast_json}

if __name__ == '__main__':
    app.run(debug=True)
