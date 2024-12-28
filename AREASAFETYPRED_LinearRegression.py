import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv('d:/Data science/Area Safety Prediction.csv')


# Independent and dependent variables for linear regression
X_linear = data.drop(columns=['outcome', 'class'])
y_linear = data['outcome']

# Splitting data for linear regression
X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(X_linear, y_linear, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train_linear, y_train_linear)
linear_predictions = linear_model.predict(X_test_linear)

print("Linear Regression coefficients:", linear_model.coef_)
print("Linear Regression intercept:", linear_model.intercept_)
