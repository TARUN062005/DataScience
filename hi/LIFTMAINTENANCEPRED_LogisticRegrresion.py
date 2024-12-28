import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 2)  # 100 data points with 2 features
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Binary target based on the sum of X1 and X2

# Convert to pandas DataFrame
data = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
data['Target'] = y

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

# Predictions
y_pred = logreg_model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# Visualization
plt.figure(figsize=(10, 6))

# Scatter plot of data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=100, label='Data Points')

# Plot the decision boundary
h = .02  # Step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = logreg_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.show()
