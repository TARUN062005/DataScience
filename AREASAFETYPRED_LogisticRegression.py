import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv('d:/Data science/Area Safety Prediction.csv')

# Independent and dependent variables for logistic regression
X_logistic = data.drop(columns=['class', 'outcome'])
y_logistic = data['class']

# Splitting data for logistic regression
X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(X_logistic, y_logistic, test_size=0.2, random_state=42)

# Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_logistic, y_train_logistic)
logistic_predictions = logistic_model.predict(X_test_logistic)

print("Logistic Regression Accuracy:", accuracy_score(y_test_logistic, logistic_predictions))
print("Classification Report:\n", classification_report(y_test_logistic, logistic_predictions))
