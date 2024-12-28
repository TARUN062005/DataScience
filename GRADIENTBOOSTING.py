import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression

# Define the GradientBoosting class
class GradientBoosting:
    def __init__(self, n_estimators=10, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []

    def fit(self, X, y):
        residuals = np.array(y)
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=3)
            tree.fit(X, residuals)
            predictions = tree.predict(X)
            residuals -= self.learning_rate * predictions
            self.trees.append(tree)
            # Debugging: print the residuals after each tree
            print(f"Iteration {i + 1}, residuals: {residuals[:5]}")  # Print first 5 residuals

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

# Create some synthetic data for testing
X, y = make_regression(n_samples=100, n_features=5, noise=0.1)

# Instantiate and train the model
model = GradientBoosting(n_estimators=10, learning_rate=0.1)
model.fit(X, y)

# Predict and print some results
predictions = model.predict(X)
print(predictions[:10])  # Print first 10 predictions
