# app.py

import streamlit as st
import pandas as pd
import joblib

# Load trained logistic regression model
model = joblib.load("models/logistic_regression_model.pkl")

# Page config
st.set_page_config(page_title="Breast Cancer Diagnosis Predictor", layout="centered")
st.title("Breast Cancer Diagnosis Predictor")

st.markdown("""
This app uses a trained **Logistic Regression** model to predict whether a tumor is **benign (0)** or **malignant (1)** 
based on diagnostic measurements from the Breast Cancer Wisconsin dataset.
""")

# Features selected with |r| > 0.4
selected_features = [
    "Radius1", "Perimeter1", "Area1",
    "Concavity1", "ConcavePoints1",
    "Radius3", "Perimeter3", "Area3",
    "Concavity3", "ConcavePoints3"
]

# Sidebar input sliders
st.sidebar.header("Enter Diagnostic Measurements")

user_input = {}
for feature in selected_features:
    user_input[feature] = st.sidebar.slider(
        label=feature,
        min_value=0.0,
        max_value=2000.0 if "Area" in feature else 100.0,
        value=10.0,
        step=0.1
    )

# Convert inputs to DataFrame
input_df = pd.DataFrame([user_input])

# Predict
if st.button("Predict Diagnosis"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"**Prediction: Malignant (1)**\n\nProbability: {probability:.2%}")
    else:
        st.success(f"**Prediction: Benign (0)**\n\nProbability: {1 - probability:.2%}")
