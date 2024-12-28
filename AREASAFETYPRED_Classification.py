import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv('d:/Data science/Area Safety Prediction.csv')

# Independent and dependent variables for classification
X_classification = data.drop(columns=['class', 'outcome'])
y_classification = data['class']

# Splitting data for classification
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Decision Tree Classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train_class, y_train_class)
classification_predictions = classifier.predict(X_test_class)

print("Classification Accuracy:", accuracy_score(y_test_class, classification_predictions))
print("Classification Report:\n", classification_report(y_test_class, classification_predictions))