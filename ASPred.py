#IMPORTS
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


#DATA HANDLING
data = pd.read_csv("Area Safety Prediction.csv")
print(data.head())
print(data.describe())
X = data[["area","sex ratio","r cases","crimes","wine shops","men literacy","porn access","psych cases","desserted area","ring roads","slum areas","season","time of visit"]]
Y = data["outcome"]
Y_ = data["class"]


#LINEAR REGRESSION
mdl = LinearRegression()
mdl.fit(X, Y)
pred = mdl.predict([[1,756,44,175,13,71,4,33,0.2,2,0.1,4,1]])
print("Predicted value (LR): ",pred[0])
print("Accuracy (LR): ",mdl.score(X[:100], Y[:100])*100)

plt.scatter(X['area'], Y, color='b')
plt.plot(X['area'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Area')  
plt.ylabel('Safety Score') 
plt.show()

#DECISION TREE CLASSIFICATION
from sklearn.tree import DecisionTreeClassifier
mdl = DecisionTreeClassifier(max_leaf_nodes=3, random_state=1)
mdl.fit(X, Y_)
pred = mdl.predict([[1,756,44,175,13,71,4,33,0.2,2,0.1,4,1]])
print("Predicted value (DTC): ",pred[0])
print("Accuracy (DTC): ",mdl.score(X[:100], Y_[:100])*100)

plt.scatter(X['area'], Y_, color='b')
plt.plot(X['area'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Area')  
plt.ylabel('Safe/Not') 
plt.show()