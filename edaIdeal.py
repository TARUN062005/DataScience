import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Ideal_Workplace_Dataset.csv'  # Update path if necessary
data = pd.read_csv(file_path)

# Step 1: Check for missing values
print("Missing values per column before handling:")
print(data.isnull().sum())

# Fill missing numerical values with the column mean
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())

# Fill missing categorical values with the mode
categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

# Step 2: Encode categorical variables
encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])

# Step 3: Normalize numerical features
scaler = MinMaxScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Step 4: Basic statistics after preprocessing
print("\nBasic statistics of the dataset after preprocessing:")
print(data.describe())

# Step 5: Visualizations
# Univariate Analysis: Histogram of a numerical feature (e.g., Employee Satisfaction Rating)
plt.figure(figsize=(8, 6))
sns.histplot(data['Employee Satisfaction Rating (1-5)'], bins=10, kde=True, color='blue')
plt.title("Distribution of Employee Satisfaction Rating")
plt.xlabel('Employee Satisfaction Rating (Normalized)')
plt.ylabel('Frequency')
plt.show()

# Bivariate Analysis: Scatter plot (e.g., Work-Life Balance vs. Average Working Hours)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='Work-Life Balance (1-5)', y='Average Working Hours', hue='Employee Satisfaction Rating (1-5)')
plt.title("Work-Life Balance vs. Average Working Hours")
plt.xlabel('Work-Life Balance (Normalized)')
plt.ylabel('Average Working Hours (Normalized)')
plt.grid(True)
plt.show()

# Multivariate Analysis: Heatmap of correlations
plt.figure(figsize=(10, 8))
correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# Step 6: Save the cleaned dataset
output_path = 'Preprocessed_Ideal_Workplace_Dataset.csv'
data.to_csv(output_path, index=False)
print(f"\nPreprocessed data saved to: {output_path}")
