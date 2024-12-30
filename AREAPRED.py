#IMPORTS
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

#DATA HANDLING
data = pd.read_csv('Area Safety Prediction.csv')
print(data.head())
print(data.describe())

X_reg = data[["sex ratio", "r cases", "crimes", "wine shops"]]  # Features for regression
Y_reg = data["outcome"]  # Target for regression

X_clf = data[["psych cases", "desserted area", "ring roads", "slum areas"]]  # Features for classification
Y_clf = data["class"]  # Target for classification

#DATA ANALYSIS
plt.scatter(X_reg["crimes"], Y_reg, color='b')
plt.xlabel('Crimes')
plt.ylabel('Outcome')
plt.show()

#LINEAR REGRESSION
mdl = LinearRegression()
mdl.fit(X_reg, Y_reg)
pred = mdl.predict(pd.DataFrame([[850, 44, 175, 13]], columns=["sex ratio", "r cases", "crimes", "wine shops"]))
print("Predicted outcome (LR): ", pred[0])
print("Accuracy (LR): ", mdl.score(X_reg, Y_reg) * 100)

plt.scatter(X_reg["crimes"], Y_reg, color='b')
plt.plot(X_reg["crimes"], mdl.predict(X_reg), color='black', linewidth=3)
plt.xlabel('Crimes')
plt.ylabel('Outcome')
plt.show()

#LOGISTIC REGRESSION
mdl = LogisticRegression()
mdl.fit(X_clf, Y_clf)
pred = mdl.predict(pd.DataFrame([[33, 0.2, 2, 0.1]], columns=["psych cases", "desserted area", "ring roads", "slum areas"]))
print("Predicted class (LGR): ", pred[0])
print("Accuracy (LGR): ", mdl.score(X_clf, Y_clf) * 100)

#SVR
mdl = SVR(kernel='rbf')
mdl.fit(X_reg, Y_reg)
pred = mdl.predict(pd.DataFrame([[850, 44, 175, 13]], columns=["sex ratio", "r cases", "crimes", "wine shops"]))
print("Predicted outcome (SVR): ", pred[0])

#SVC
mdl = SVC(kernel='poly')
mdl.fit(X_clf, Y_clf)
pred = mdl.predict(pd.DataFrame([[33, 0.2, 2, 0.1]], columns=["psych cases", "desserted area", "ring roads", "slum areas"]))
print("Predicted class (SVC): ", pred[0])

#NAIVE BAYES
mdl = GaussianNB()
mdl.fit(X_clf, Y_clf)
pred = mdl.predict(pd.DataFrame([[33, 0.2, 2, 0.1]], columns=["psych cases", "desserted area", "ring roads", "slum areas"]))
print("Predicted class (NB): ", pred[0])

#KNN
mdl = KNeighborsClassifier()
mdl.fit(X_clf, Y_clf)
pred = mdl.predict(pd.DataFrame([[33, 0.2, 2, 0.1]], columns=["psych cases", "desserted area", "ring roads", "slum areas"]))
print("Predicted class (KNN): ", pred[0])

#RANDOM FOREST
mdl = RandomForestClassifier()
mdl.fit(X_clf, Y_clf)
pred = mdl.predict(pd.DataFrame([[33, 0.2, 2, 0.1]], columns=["psych cases", "desserted area", "ring roads", "slum areas"]))
print("Predicted class (RFC): ", pred[0])

mdl = RandomForestRegressor()
mdl.fit(X_reg, Y_reg)
pred = mdl.predict(pd.DataFrame([[850, 44, 175, 13]], columns=["sex ratio", "r cases", "crimes", "wine shops"]))
print("Predicted outcome (RFR): ", pred[0])

#DECISION TREE
mdl = DecisionTreeClassifier()
mdl.fit(X_clf, Y_clf)
pred = mdl.predict(pd.DataFrame([[33, 0.2, 2, 0.1]], columns=["psych cases", "desserted area", "ring roads", "slum areas"]))
print("Predicted class (DTC): ", pred[0])

mdl = DecisionTreeRegressor()
mdl.fit(X_reg, Y_reg)
pred = mdl.predict(pd.DataFrame([[850, 44, 175, 13]], columns=["sex ratio", "r cases", "crimes", "wine shops"]))
print("Predicted outcome (DTR): ", pred[0])

#KMEANS
k = 3
mdl = KMeans(n_clusters=k)
mdl.fit(X_clf)
centroids = mdl.cluster_centers_
print("Centroids: ", centroids)
pred = mdl.predict(pd.DataFrame([[33, 0.2, 2, 0.1]], columns=["psych cases", "desserted area", "ring roads", "slum areas"]))
print("Predicted cluster (KM): ", pred[0])
