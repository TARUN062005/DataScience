import pandas as pd

df = pd.read_csv('LiftMaintainancePrediction.csv')

data = df[['p','q','r']]

print(data.head())

print(data.describe())