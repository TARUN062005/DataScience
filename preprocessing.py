# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('LiftMaintainancePrediction.csv')
print(df.head())
print(df.describe())

# Step 1: Handling Missing Values
# Fill missing values for numeric fields
df['TEMPERATURE'] = df['TEMPERATURE'].fillna(df['TEMPERATURE'].mean())
df['CAPACITY'] = df['CAPACITY'].fillna(df['CAPACITY'].median())
df['SPEED'] = df['SPEED'].interpolate()  # Interpolation for SPEED

# Fill missing values for categorical fields
df['CLASS'] = df['CLASS'].fillna(df['CLASS'].mode()[0])

# Step 2: Encoding Categorical Variables
# One-Hot Encoding for 'CLASS' column
df = pd.get_dummies(df, columns=['CLASS'], prefix='CLASS', drop_first=True)

# Step 3: Scaling and Normalization
# Select numeric columns for scaling
numeric_cols = ['TEMPERATURE', 'CAPACITY', 'SPEED', 'DOOR_OPENING_TIME', 'p', 'q', 'r']

# Apply Min-Max Scaling
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Step 4: Feature Engineering
# Create new features
df['Volume'] = df['CAPACITY'] * df['SPEED']
df['Door_Efficiency'] = df['DOOR_OPENING_TIME'] / (df['SPEED'] + 1e-6)  # Avoid division by zero

# Display the first few rows of the preprocessed dataset
print(df.head())
print(df.describe())

df.to_csv('LiftMaintainancePredictionProcessed.csv', index=False)