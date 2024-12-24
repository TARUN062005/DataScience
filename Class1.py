import pandas as pd

df=pd.read_csv('ann.csv ')
#data=df['p','q','r']
 
# print(data.head())
print(df.head())

print(df.describe())