import numpy as np

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
            new_centroids = np.array([
                X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else self.centroids[i]
                for i in range(self.k)
            ])
            
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

# Example Usage
X = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
    [1.0, 0.6],
    [9.0, 11.0]
])

# Initialize the KMeansScratch model
model = KMeansScratch(n_clusters=3, max_iters=100)

# Fit the model to the dataset
model.fit(X)

# Predict cluster labels
labels = model.predict(X)

# Print results
print("Centroids:", model.centroids)
print("Labels:", labels)
