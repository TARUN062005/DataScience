import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
# Load a dataset 
df = pd.read_csv('LiftMaintainancePrediction.csv')
# Univariate Analysis: Histogram 
sns.histplot(df['TEMPERATURE'], bins=30, kde=True) 
plt.title("Distribution of Temperature") 
plt.show() 
# Bivariate Analysis: Scatter Plot 
sns.scatterplot(data=df, x='CAPACITY', y='TEMPERATURE', hue='SPEED') 
plt.title("CAPACITY vs TEMPERATURE") 
plt.show() 
# Multivariate Analysis: Heatmap 
correlation = df.corr() 
sns.heatmap(correlation, annot=True, cmap='coolwarm') 
plt.title("Correlation Heatmap") 
plt.show() 