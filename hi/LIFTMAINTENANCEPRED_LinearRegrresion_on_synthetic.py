import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(42)  # For reproducibility
X = 2 * np.random.rand(100, 1)  # 100 random points between 0 and 2
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relation with noise

# Convert to a pandas DataFrame for easier manipulation
data = pd.DataFrame(data=np.column_stack([X, y]), columns=["X", "y"])

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predictions
y_pred = linear_model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Regression Coefficients:", linear_model.coef_)
print("Intercept:", linear_model.intercept_)

# Visualization
plt.figure(figsize=(10, 6))

# Scatter plot of the data
plt.scatter(X, y, color='blue', label='Data Points')

# Regression line
plt.plot(X, linear_model.predict(X), color='red', linewidth=2, label='Regression Line')

# Add labels and title
plt.title('Linear Regression on Synthetic Data')
plt.xlabel('X (Independent Variable)')
plt.ylabel('y (Dependent Variable)')
plt.legend()

# Show plot
plt.show()
