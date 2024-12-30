#IMPORTS
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


#DATA HANDLING
data = pd.read_csv('prices.csv')
print(data.head())
print(data.describe())
X = data[["Rooms"]]
Y = data["Value"]
Y_ = data["Class"]


#DATA ANALYSIS
plt.scatter(X['Rooms'], Y, color='b')
plt.xlabel('Rooms')  
plt.ylabel('Value') 
plt.show()


#OBSERVATIONS
print("The highest number of houses has 5.5 to 6.5 rooms with the price in between 12 and 30.")


#LINEAR REGRESSION
mdl = LinearRegression()
mdl.fit(X, Y)
pred = mdl.predict([[6.575]])
print("Predicted value (LR): ",pred[0])
print("Accuracy (LR): ",mdl.score(X[:100], Y[:100])*100)

plt.scatter(X['Rooms'], Y, color='b')
plt.plot(X['Rooms'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Rooms')  
plt.ylabel('Value') 
plt.show()


#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
mdl = LogisticRegression()
mdl.fit(X, Y_)
pred = mdl.predict([[6.575]])
print("Predicted value (LGR): ",pred[0])
print("Accuracy (LGR): ",mdl.score(X[:100], Y_[:100])*100)

plt.scatter(X['Rooms'], Y_, color='b')
plt.plot(X['Rooms'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Rooms')  
plt.ylabel('Value')
plt.show()


#SVR
from sklearn.svm import SVR
mdl = SVR(kernel = 'rbf')
mdl.fit(X, Y)
pred = mdl.predict([[6.575]])
print("Predicted value (SVR): ",pred[0])
print("Accuracy (SVR): ",mdl.score(X[:100], Y[:100])*100)

plt.scatter(X['Rooms'], Y, color='b')
plt.plot(X['Rooms'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Rooms')
plt.ylabel('Value')
plt.show()



#SVC
from sklearn.svm import SVC
mdl = SVC(kernel='poly')
mdl.fit(X, Y_)
pred = mdl.predict([[6.575]])
print("Predicted value (SVC): ",pred[0])
print("Accuracy (SVC): ",mdl.score(X[:100], Y_[:100])*100)

plt.scatter(X['Rooms'], Y_, color='b')
plt.plot(X['Rooms'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Rooms')  
plt.ylabel('High/Low') 
plt.show()


#NB
from sklearn.naive_bayes import GaussianNB
mdl = GaussianNB()
mdl.fit(X, Y_)
pred = mdl.predict([[6.575]])
print("Predicted value (NB): ",pred[0])
print("Accuracy (NB): ",mdl.score(X[:100], Y_[:100])*100)

plt.scatter(X['Rooms'], Y_, color='b')
plt.plot(X['Rooms'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Rooms')  
plt.ylabel('High/Low') 
plt.show()


#KNN
from sklearn.neighbors import KNeighborsClassifier
mdl = KNeighborsClassifier()
mdl.fit(X, Y_)
pred = mdl.predict([[6.575]])
print("Predicted value (KNN): ",pred[0])
print("Accuracy (KNN): ",mdl.score(X[:100], Y_[:100])*100)

plt.scatter(X['Rooms'], Y_, color='b')
plt.plot(X['Rooms'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Rooms')  
plt.ylabel('High/Low') 
plt.show()


#RANDOM FOREST CLASSIFICATION
from sklearn.ensemble import RandomForestClassifier
mdl = RandomForestClassifier(criterion='entropy')
mdl.fit(X, Y_)
pred = mdl.predict([[6.575]])
print("Predicted value (RFC): ",pred[0])
print("Accuracy (RFC): ",mdl.score(X[:100], Y_[:100])*100)

plt.scatter(X['Rooms'], Y_, color='b')
plt.plot(X['Rooms'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Rooms')  
plt.ylabel('High/Low') 
plt.show()


#RANDOM FOREST REGRESSION
from sklearn.ensemble import RandomForestRegressor
mdl = RandomForestRegressor(n_estimators=100,max_depth=6)
mdl.fit(X, Y)
pred = mdl.predict([[6.575]])
print("Predicted value (RFR): ",pred[0])
print("Accuracy (RFR): ",mdl.score(X[:100], Y[:100])*100)

plt.scatter(X['Rooms'], Y, color='b')
plt.plot(X['Rooms'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Rooms')  
plt.ylabel('High/Low') 
plt.show()


#DECISION TREE CLASSIFICATION
from sklearn.tree import DecisionTreeClassifier
mdl = DecisionTreeClassifier(max_leaf_nodes=3, random_state=1)
mdl.fit(X, Y_)
pred = mdl.predict([[6.575]])
print("Predicted value (DTC): ",pred[0])
print("Accuracy (DTC): ",mdl.score(X[:100], Y_[:100])*100)

plt.scatter(X['Rooms'], Y_, color='b')
plt.plot(X['Rooms'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Rooms')  
plt.ylabel('High/Low') 
plt.show()


#DECISION TREE REGRESSION
from sklearn.tree import DecisionTreeRegressor
mdl =  DecisionTreeRegressor(max_depth=3)
mdl.fit(X, Y)
pred = mdl.predict([[6.575]])
print("Predicted value (DTR): ",pred[0])
print("Accuracy (DTR): ",mdl.score(X[:100], Y[:100])*100)

plt.scatter(X['Rooms'], Y, color='b')
plt.plot(X['Rooms'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Rooms')  
plt.ylabel('High/Low') 
plt.show()


#KMEANS
from sklearn.cluster import KMeans
k = 4
mdl = KMeans(n_clusters=k)
mdl.fit(data.iloc[1:])
centroids = mdl.cluster_centers_
print("Centroids: ",centroids)
pred = mdl.predict([[6.575,4.09,24,0]])
print("Predicted value (KM): ",pred[0])

labels = mdl.labels_
colors = ['blue','red','green','black','purple']
y = 0
for x in labels:
    # plot the points acc to their clusters
    # and assign different colors
    plt.scatter(data.iloc[y,0], data.iloc[y,1],color=colors[x])
    y+=1
        
for x in range(k):
    #plot the centroids
    lines = plt.plot(centroids[x,0],centroids[x,1],'kx')    
    #make the centroid larger    
    plt.setp(lines,ms=15.0)
    plt.setp(lines,mew=2.0)
    
title = ('No of clusters (k) = {}').format(k)
plt.title(title)
plt.xlabel('eruptions (mins)')
plt.ylabel('waiting (mins)')
plt.show()


#MULTIPLE LINEAR REGRESSION
X = data[["Rooms","Distance"]]
mdl = LinearRegression()
mdl.fit(X, Y)
pred = mdl.predict([[6.575, 4.09]])
print("Predicted value (MLR): ",pred[0])
print("Accuracy (MLR): ",mdl.score(X[:100], Y[:100])*100)

plt.scatter(X['Rooms'], Y, color='b')
plt.plot(X['Rooms'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Rooms')  
plt.ylabel('Value') 
plt.show()