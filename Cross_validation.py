import numpy as np

def k_fold_cross_validation(X, y, k=5):
    """
    Perform k-fold cross-validation by splitting the dataset into k folds.
    
    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target array.
        k (int): Number of folds. Default is 5.
        
    Returns:
        list: List of arrays, where each array contains indices for one fold.
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)  # Shuffle indices to ensure random splitting
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1  # Distribute leftover samples among the first folds
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
    return folds

# Example dataset
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
    [9, 10],
    [11, 12],
    [13, 14],
    [15, 16],
    [17, 18],
    [19, 20]
])
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

# Perform k-fold cross-validation
k = 5
folds = k_fold_cross_validation(X, y, k=k)

# Example model training and testing loop
for i in range(k):
    test_idx = folds[i]
    train_idx = np.hstack([folds[j] for j in range(k) if j != i])
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"Fold {i + 1}")
    print("Train indices:", train_idx)
    print("Test indices:", test_idx)
    print("X_train:\n", X_train)
    print("X_test:\n", X_test)
    print("y_train:", y_train)
    print("y_test:", y_test)
    print()
    # Train your model using X_train, y_train
    # Evaluate your model on X_test, y_test
