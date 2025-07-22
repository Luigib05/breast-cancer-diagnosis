# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("model.joblib")

# Set up the page
st.set_page_config(page_title="Breast Cancer Diagnosis Predictor", layout="centered")

st.title("Breast Cancer Diagnosis Predictor")
st.markdown("""
This app uses a trained **Logistic Regression** model to predict whether a tumor is **benign (0)** or **malignant (1)** 
based on diagnostic features from the Breast Cancer Wisconsin dataset.
""")

# Define the input features
input_features = [
    "radius1", "texture1", "perimeter1", "area1", "smoothness1",
    "compactness1", "concavity1", "concave_points1", "symmetry1", "fractal_dimension1"
]

# Collect input values
st.sidebar.header("Input Diagnostic Measurements")
user_input = {}

for feature in input_features:
    user_input[feature] = st.sidebar.slider(
        label=feature,
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.1
    )

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Predict
if st.button("Predict Diagnosis"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"Prediction: **Malignant (1)** \n\nProbability: {probability:.2%}")
    else:
        st.success(f"Prediction: **Benign (0)** \n\nProbability: {1 - probability:.2%}")
