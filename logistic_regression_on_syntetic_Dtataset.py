import numpy as np
import matplotlib.pyplot as plt

# Define Logistic Regression from scratch
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None  # Parameters (weights)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        # Add bias term (intercept) as the first column in X
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        m = len(X_b)
        self.theta = np.random.randn(X_b.shape[1])  # Random initialization of theta
        
        for _ in range(self.n_iterations):
            # Compute predictions (probabilities)
            predictions = self.sigmoid(X_b.dot(self.theta))
            # Compute the gradient
            gradients = (1 / m) * X_b.T.dot(predictions - y)
            # Update the weights using gradient descent
            self.theta -= self.learning_rate * gradients
    
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        predictions = self.sigmoid(X_b.dot(self.theta))
        return (predictions >= 0.5).astype(int)

# Evaluation metrics
def precision_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0

def recall_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0

def f1_score(y_true, y_pred):
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0

# Generate synthetic data for Logistic Regression
np.random.seed(42)
X_pos = np.random.randn(50, 2) + np.array([2, 2])  # Class 1
X_neg = np.random.randn(50, 2) + np.array([-2, -2])  # Class 0
X = np.vstack((X_pos, X_neg))
y = np.array([1]*50 + [0]*50)

# Initialize and train the model
model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)
predictions = model.predict(X)

# Plot the results
plt.scatter(X[y==1][:,0], X[y==1][:,1], color='blue', label='Class 1')
plt.scatter(X[y==0][:,0], X[y==0][:,1], color='red', label='Class 0')
plt.scatter(X[:,0], X[:,1], c=predictions, cmap='bwr', alpha=0.3, label='Predicted')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Calculate evaluation metrics
precision = precision_score(y, predictions)
recall = recall_score(y, predictions)
f1 = f1_score(y, predictions)

print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
