import numpy as np
from sklearn.tree import DecisionTreeClassifier  # Use DecisionTreeClassifier for classification

class RandomForest:
    def __init__(self, n_trees=10, max_depth=3, bootstrap=True):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=self.bootstrap)
        return X[indices], y[indices]

    def predict(self, X):
        # Collect predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        # Majority voting for classification
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_predictions)

# Example Usage
if __name__ == "__main__":
    # Example dataset (replace this with your actual data)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    
    forest = RandomForest(n_trees=10, max_depth=3)
    forest.fit(X, y)
    predictions = forest.predict(X)
    print("Predictions:", predictions)
