import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset (assuming you have it as a CSV)
data = pd.read_csv("Area Safety Prediction.csv")

# Display the first few rows of the dataset
print(data.head())

# Separate features (X) and target variable (y)
X = data.drop(columns=["class"])
y = data["class"]

# Handle missing values (if any)
imputer = SimpleImputer(strategy="mean")  # Replace missing values with the mean of the column
X_imputed = imputer.fit_transform(X)

# Feature scaling: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a model (Random Forest classifier as an example)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

