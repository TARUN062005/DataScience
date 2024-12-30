import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        covariance_matrix = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        eigenvectors = eigenvectors.T
        indices = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[indices[:self.n_components]]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)

# Corrected usage of variable name
x = np.random.rand(100, 5)
pca = PCA(n_components=2)
pca.fit(x)
X_reduced = pca.transform(x)

# Print the reduced data
print("Reduced Data (first 5 samples):")
print(X_reduced[:5])
