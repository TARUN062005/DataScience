import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('d:/Data science/Area Safety Prediction.csv')

# Feature scaling
scaler = StandardScaler()
X_cluster = scaler.fit_transform(data.drop(columns=['outcome', 'class']))

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_cluster)
data['Cluster'] = kmeans.labels_

# Visualizing Clusters (using the first two features as an example)
plt.figure(figsize=(8, 6))
plt.scatter(X_cluster[:, 0], X_cluster[:, 1], c=kmeans.labels_, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('Clustering Visualization')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.legend()
plt.show()

print("Cluster centers:\n", kmeans.cluster_centers_)
print("Cluster assignments:\n", data['Cluster'].value_counts())
