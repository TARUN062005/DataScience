import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load the dataset
file_path = 'Ideal_Workplace_Dataset.csv'  # Update path if needed
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

# Step 5: Save the cleaned dataset
output_path = 'Preprocessed_Ideal_Workplace_Dataset.csv'
data.to_csv(output_path, index=False)
print(f"\nPreprocessed data saved to: {output_path}")
