import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

st.title("🎯 Decision Tree Classifier - Social Network Ads")

# Load dataset
data = pd.read_csv(r"C:\Users\rapol\Downloads\DATA SCIENCE\4. Dec\11th - SVM\Social_Network_Ads.csv")

# Show the data
if st.checkbox("Show Dataset"):
    st.write(data.head())

# Select features and target
x = data.iloc[:, [2, 3]].values  # Age and Estimated Salary
y = data.iloc[:, -1].values      # Purchased

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# Train model
dt = DecisionTreeClassifier(random_state=0)
dt.fit(x_train, y_train)

# Predict
y_pred = dt.predict(x_test)

# Metrics
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
bias = dt.score(x_test, y_test)
variance = dt.score(x_train, y_train)

# Display Results
st.subheader("📊 Model Evaluation")
st.write("**Accuracy Score:**", accuracy)
st.write("**Bias (Test Accuracy):**", bias)
st.write("**Variance (Train Accuracy):**", variance)
st.write("**Confusion Matrix:**")
st.write(cm)

# Visualization
st.subheader("🧠 Decision Boundary (Training Set)")

# Plot only if checkbox selected
if st.checkbox("Show Decision Boundary"):
    X_set, y_set = x_train, y_train
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=1),
        np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=1000)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contourf(X1, X2, dt.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha=0.3, cmap=ListedColormap(('red', 'green')))
    ax.scatter(X_set[:, 0], X_set[:, 1], c=y_set, cmap=ListedColormap(('red', 'green')))
    ax.set_title('Decision Tree Classifier (Training Set)')
    ax.set_xlabel('Age')
    ax.set_ylabel('Estimated Salary')
    st.pyplot(fig)
