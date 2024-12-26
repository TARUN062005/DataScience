import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv("Area Safety Prediction.csv")

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Plot histograms for each feature with a line (e.g., density line) overlaid
# We will use seaborn's histplot with kde (kernel density estimate) to show a line on the histogram
for column in data.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[column], kde=True, bins=20)  # kde=True adds a density line
    plt.title(f'Histogram of {column} with Density Line')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Scatter plot with multiple colors based on a categorical feature (e.g., 'class')
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['sex ratio'], y=data['crimes'], hue=data['class'], palette='viridis')
plt.title('Scatter Plot between Sex Ratio and Crimes with Class as Hue')
plt.xlabel('Sex Ratio')
plt.ylabel('Crimes')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))  # Set figure size for the heatmap
corr = data.corr()  # Compute pairwise correlation of columns
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=16)
plt.show()

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
