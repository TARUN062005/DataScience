import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
import numpy as np

# Title of the App
st.title("MULTIMODAL ML APP with Preloaded Dataset")

# Preload the dataset
data = pd.read_csv("Area Safety Prediction.csv")

st.write("Preview of the dataset:")
st.write(data.head())

# Step 2: Select Features and Target
st.write("Step 2: Define Features and Target")
columns = data.columns.tolist()

# Selecting features without pre-selecting them
feature_cols = st.multiselect("Select Feature Columns", columns)

# Select target column
target_col = st.selectbox("Select Target Column", columns)

if feature_cols and target_col:
    X = data[feature_cols]
    Y = data[target_col]

    # Step 3: Input Data for Prediction
    st.write("Step 3: Input Data for Prediction")
    inputs = {}
    for col in feature_cols:
        inputs[col] = st.number_input(f"Input {col}", value=float(data[col].mean()))
    input_data = np.array([list(inputs.values())])

    # Step 4: Choose Algorithm
    st.write("Step 4: Choose Algorithm")
    algorithms = {
        "Linear Regression": LinearRegression(),
        "Logistic Regression": LogisticRegression(),
        "Support Vector Regression (SVR)": SVR(kernel='rbf'),
        "Support Vector Classifier (SVC)": SVC(kernel='poly'),
        "Naive Bayes": GaussianNB(),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
        "Random Forest Classifier": RandomForestClassifier(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "K-Means Clustering": KMeans(n_clusters=4)
    }
    algorithm = st.selectbox("Select an Algorithm", list(algorithms.keys()))

    # Scale features for algorithms like KNN or SVR
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if st.button("Predict"):
        model = algorithms[algorithm]
        if "Classifier" in algorithm or "Regression" in algorithm:
            model.fit(X_scaled, Y)
            prediction = model.predict(input_data)
            st.write(f"Prediction: {prediction[0]}")
            if "Regressor" in algorithm:
                st.write(f"R2 Score: {r2_score(Y, model.predict(X_scaled)) * 100:.2f}%")
                st.write(f"Mean Squared Error: {mean_squared_error(Y, model.predict(X_scaled)):.2f}")
            else:
                st.write(f"Accuracy: {model.score(X_scaled, Y) * 100:.2f}%")

            # Visualize Results
            st.write("Visualization:")
            plt.scatter(X_scaled[:, 0], Y, color='b', label='Data')
            plt.plot(X_scaled[:, 0], model.predict(X_scaled), color='r', label='Prediction')
            plt.xlabel(feature_cols[0])
            plt.ylabel(target_col)
            plt.legend()
            st.pyplot(plt)

        elif "K-Means" in algorithm:
            model.fit(X_scaled)
            st.write(f"Cluster Centers: {model.cluster_centers_}")
            prediction = model.predict(input_data)
            st.write(f"Prediction: Cluster {prediction[0]}")

            # Plot Clusters
            labels = model.labels_
            colors = ['blue', 'red', 'green', 'black', 'purple']
            plt.figure()
            for idx, label in enumerate(labels):
                plt.scatter(X_scaled[idx, 0], X_scaled[idx, 1], color=colors[label])
            for center in model.cluster_centers_:
                plt.scatter(center[0], center[1], color='yellow', s=200, marker='X', edgecolor='black')
            st.pyplot(plt)
