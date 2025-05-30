import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the data
df = pd.read_csv('/Users/macbook/arima_forcast/sample_customer_data.csv')

# Step 2: Encode categorical features
label_encoders = {}
for col in ['FavDish', 'TimePref', 'Location']:
    le = LabelEncoder()
    df[col + '_Encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 3: Feature selection
features = ['TotalSpend', 'Frequency', 'AOV', 'FavDish_Encoded', 'TimePref_Encoded', 'Location_Encoded']
X = df[features]

# Step 4: Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 6: Analyze cluster profiles
cluster_profiles = df.groupby('Cluster').agg({
    'TotalSpend': 'mean',
    'Frequency': 'mean',
    'AOV': 'mean',
    'FavDish': lambda x: x.value_counts().idxmax(),
    'TimePref': lambda x: x.value_counts().idxmax(),
    'Location': lambda x: x.value_counts().idxmax()
}).reset_index()

print("\n📊 Cluster Profiles:\n")
print(cluster_profiles)

# Step 7: Assign promotions manually based on cluster
promotion_map = {
    0: "Loyalty Points + 'Come Back Soon' Offer",
    1: "Upsell Combo Deals for Regulars",
    2: "Lunch-Time Discount Bundle",
    3: "Reactivation Coupon - Limited Time Offer"
}
df['Promotion'] = df['Cluster'].map(promotion_map)

# Step 8: Save full customer + cluster + promotion data
df.to_csv('/Users/macbook/arima_forcast/customers_with_clusters.csv', index=False)
cluster_profiles.to_csv('/Users/macbook/arima_forcast/cluster_profiles.csv', index=False)

# Step 9: Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='TotalSpend', y='Frequency', hue='Cluster', palette='Set2')
plt.title('Customer Segments: Total Spend vs Frequency')
plt.xlabel('Total Spend')
plt.ylabel('Frequency')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig('/Users/macbook/arima_forcast/cluster_scatterplot.png')
plt.show()

# Step 10: Visualize promotion assignments
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Promotion', order=df['Promotion'].value_counts().index, palette='Set3')
plt.title("Promotion Assignment per Cluster")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('/Users/macbook/arima_forcast/promotion_distribution.png')
plt.show()

# ✅ Step 11: Save customer IDs by cluster for promotion targeting
cluster_groups = df.groupby('Cluster')['CustomerID'].apply(list).reset_index()
cluster_groups.columns = ['Cluster', 'CustomerIDs']
cluster_groups.to_csv('/Users/macbook/arima_forcast/customer_ids_by_cluster.csv', index=False)

print("\n✅ Output saved:")
print("- customers_with_clusters.csv")
print("- cluster_profiles.csv")
print("- cluster_scatterplot.png")
print("- promotion_distribution.png")
print("- customer_ids_by_cluster.csv")
