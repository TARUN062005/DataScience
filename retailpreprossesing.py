import pandas as pd

# Load the dataset
file_path = 'Retail_Sales_Data.csv'  # Update path if necessary
data = pd.read_csv(file_path)

# Step 1: Remove unnecessary columns
data = data.drop(columns=['Unnamed: 1'], errors='ignore')

# Step 2: Clean the data
# Ensure Total Sales equals Quantity Sold * Unit Price
data['Calculated Total Sales'] = data['Quantity Sold'] * data['Unit Price']
data['Total Sales Mismatch'] = data['Total Sales'] != data['Calculated Total Sales']
if data['Total Sales Mismatch'].any():
    print("Warning: Total Sales mismatch detected. Correcting values.")
    data['Total Sales'] = data['Calculated Total Sales']
data = data.drop(columns=['Calculated Total Sales', 'Total Sales Mismatch'], errors='ignore')

# Remove duplicates
data = data.drop_duplicates()

# Step 3: Feature Engineering
# Extract City and State from Location
data[['City', 'State']] = data['Location'].str.split(',', expand=True)
data['City'] = data['City'].str.strip()
data['State'] = data['State'].str.strip()

# Step 4: Basic Statistics
stats = data.describe(include='all')

# Save the cleaned data
output_path = 'Cleaned_Retail_Sales_Data.csv'
data.to_csv(output_path, index=False)

print("Data preprocessing complete. Cleaned file saved to:", output_path)
print("Basic statistics:")
print(stats)
