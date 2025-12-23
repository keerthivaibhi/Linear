import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


st.title("Social Network Ads - KNN Demo")

# 1. Load data
uploaded = st.file_uploader("Upload Social_Network_Ads.csv", type=["csv"])
if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.write("Preview of data:")
    st.dataframe(data.head())

    # 2. Features & target
    X = data[["Age", "EstimatedSalary"]]
    y = data["Purchased"]

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    # 4. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. KNN model
    k = st.slider("Number of neighbors (k)", 1, 30, 10)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)

    # 6. Show accuracy
    acc = model.score(X_test_scaled, y_test)
    st.write(f"Test accuracy: {acc:.3f}")

    # 7. User input for prediction
    st.subheader("Try a new person")
    age = st.number_input("Age", min_value=18, max_value=70, value=45)
    salary = st.number_input("Estimated Salary", min_value=15000, max_value=200000, value=42000)

    if st.button("Predict"):
        new_point = scaler.transform([[age, salary]])
        pred = model.predict(new_point)[0]
        label = "Purchased (1)" if pred == 1 else "Not Purchased (0)"
        st.success(f"Prediction: {label}")

    # 8. Visualization
    st.subheader("Decision boundary visualization")

    show_plot = st.checkbox("Show KNN decision boundary", value=True)
    if show_plot:
        # meshgrid in original feature space
        x_min, x_max = X["Age"].min() - 1, X["Age"].max() + 1
        y_min, y_max = X["EstimatedSalary"].min() - 5000, X["EstimatedSalary"].max() + 5000
        h_x = 0.25
        h_y = 2000

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h_x),
            np.arange(y_min, y_max, h_y),
        )

        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_scaled = scaler.transform(grid)
        Z = model.predict(grid_scaled).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(7, 5))
        # decision regions
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlGn")

        # training points
        ax.scatter(
            X["Age"][y == 0],
            X["EstimatedSalary"][y == 0],
            c="red",
            s=15,
            label="Not Purchased",
        )
        ax.scatter(
            X["Age"][y == 1],
            X["EstimatedSalary"][y == 1],
            c="green",
            s=15,
            label="Purchased",
        )

        ax.set_xlabel("Age")
        ax.set_ylabel("Estimated Salary")
        ax.set_title("KNN: Social Network Ads")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

else:
    st.info("Please upload the CSV file to begin.")
