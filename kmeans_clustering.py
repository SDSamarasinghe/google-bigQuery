import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV file
df = pd.read_csv('/Users/macbook/arima_forcast/sample_customer_data.csv')

# Encode categorical columns
label_encoders = {}
for col in ['FavDish', 'TimePref', 'Location']:
    le = LabelEncoder()
    df[col + '_Encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

# Select features for clustering
features = ['TotalSpend', 'Frequency', 'AOV', 'FavDish_Encoded', 'TimePref_Encoded', 'Location_Encoded']
X = df[features]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Save clustered data
df.to_csv('clustered_customers.csv', index=False)

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='TotalSpend', y='Frequency', hue='Cluster', palette='Set2')
plt.title('Customer Segments: Total Spend vs Frequency')
plt.xlabel('Total Spend')
plt.ylabel('Frequency')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig('customer_segments_plot.png')
plt.show()
