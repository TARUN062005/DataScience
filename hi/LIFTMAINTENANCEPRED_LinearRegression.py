import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('D:/Data science/LiftMaintainancePrediction.csv')

# Independent and dependent variables
X = data.drop(columns=['OUTCOMES'])  # Replace 'OUTCOMES' with the actual target column
y = data['OUTCOMES']  # Replace 'OUTCOMES' with the actual target column

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predictions
y_pred = linear_model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Regression Coefficients:", linear_model.coef_)
print("Intercept:", linear_model.intercept_)

# Visualization (for single feature regression)
if X.shape[1] == 1:  # Only plot if there's a single feature
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X, linear_model.predict(X), color='red', linewidth=2, label='Regression Line')
    plt.title('Linear Regression')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.legend()
    plt.show()
