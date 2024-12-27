import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Define KMeans from scratch
class KMeansScratch:
    def __init__(self, n_clusters=3, max_iters=100, tolerance=1e-4):
        self.k = n_clusters
        self.max_iters = max_iters
        self.tol = tolerance
        self.centroids = None

    def fit(self, X):
        n_samples, n_features = X.shape
        # Initialize centroids randomly from the data points
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # Assign clusters
            distances = self._compute_distances(X)
            labels = np.argmin(distances, axis=1)

            # Compute new centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else self.centroids[i] for i in range(self.k)])

            # Check for convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids

    def _compute_distances(self, X):
        distances = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            distances[:, i] = np.linalg.norm(X - self.centroids[i], axis=1)
        return distances

    def predict(self, X):
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

# Generate synthetic data for K-Means
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Initialize and fit K-Means
kmeans = KMeansScratch(n_clusters=4, max_iters=100)
kmeans.fit(X)
labels = kmeans.predict(X)

# Plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.legend()
plt.show()
