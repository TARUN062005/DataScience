import numpy as np
import matplotlib.pyplot as plt

# Define Linear Regression from scratch
class LinearRegressionScratch:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None  # Parameters (weights)
    
    def fit(self, X, y):
        # Add bias term (intercept) as the first column in X
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        m = len(X_b)
        self.theta = np.random.randn(X_b.shape[1])  # Random initialization of theta
        
        for _ in range(self.n_iterations):
            gradients = (2 / m) * X_b.T.dot(X_b.dot(self.theta) - y)
            self.theta -= self.learning_rate * gradients
    
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        return X_b.dot(self.theta)

# Evaluation metrics
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# Generate synthetic data for Linear Regression
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X.flatten() + np.random.randn(100)

# Initialize and train the model
model = LinearRegressionScratch(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)
predictions = model.predict(X)

# Plot the results
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, predictions, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Calculate evaluation metrics
mse = mean_squared_error(y, predictions)
rmse = root_mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print(f"MSE: {mse}, RMSE: {rmse}, R2: {r2}")
