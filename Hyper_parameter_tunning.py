import numpy as np
import itertools

def k_fold_cross_validation(X, y, k=5):
    """
    Perform k-fold cross-validation by splitting the dataset into k folds.
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
    return folds

def grid_search(X, y, model_class, param_grid, k=5):
    """
    Perform a grid search to find the best hyperparameters for a model.
    
    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target array.
        model_class (class): Class of the model to train.
        param_grid (dict): Dictionary of hyperparameters and their values.
        k (int): Number of folds for cross-validation. Default is 5.
        
    Returns:
        tuple: Best parameters and corresponding score.
    """
    folds = k_fold_cross_validation(X, y, k)
    param_combinations = list(itertools.product(*param_grid.values()))
    best_score = -np.inf
    best_params = None

    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))
        scores = []

        for i in range(k):
            test_idx = folds[i]
            train_idx = np.hstack([folds[j] for j in range(k) if j != i])
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = model_class(**param_dict)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            score = evaluate_metric(y_test, predictions)  # Define your evaluation metric
            scores.append(score)

        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_params = param_dict

    return best_params, best_score

# Example Usage
# Define a sample model class
class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        # Dummy implementation
        pass

    def predict(self, X):
        # Dummy implementation
        return np.zeros(X.shape[0])

# Define an evaluation metric
def evaluate_metric(y_true, y_pred):
    return -np.mean((y_true - y_pred) ** 2)  # Mean Squared Error as an example

# Example dataset
X = np.random.rand(100, 2)
y = np.random.rand(100)

# Define a parameter grid
param_grid = {
    'learning_rate': [0.01, 0.1],
    'n_iterations': [1000, 5000]
}

# Perform grid search
best_params, best_score = grid_search(X, y, LinearRegressionScratch, param_grid, k=5)

print("Best Parameters:", best_params)
print("Best Score:", best_score)
