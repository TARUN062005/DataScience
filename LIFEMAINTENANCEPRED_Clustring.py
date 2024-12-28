import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic data
np.random.seed(42)
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# KMeans Clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Get the cluster centers
centers = kmeans.cluster_centers_

# Predict the cluster labels for each point
y_kmeans = kmeans.predict(X)

# Visualization
plt.figure(figsize=(8, 6))

# Scatter plot of data points colored by cluster
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Plot the cluster centers
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Cluster Centers')

# Titles and labels
plt.title('K-Means Clustering', fontsize=14)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.legend()

# Show the plot
plt.show()
