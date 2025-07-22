import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(df):
    """
    Preprocess the Breast Cancer dataset:
    - Keep only features with Pearson correlation |r| > 0.4 with diagnosis
    - Encode diagnosis labels
    - Scale features
    - Split into train and test sets

    Returns:
        X_train, X_test, y_train, y_test
    """

    # Keep only selected features (from previous correlation analysis)
    selected_features = [
        "Radius1", "Perimeter1", "Area1",
        "Concavity1", "ConcavePoints1", "Radius3",
        "Perimeter3", "Area3", "Concavity3", "ConcavePoints3"
    ]

    # Select features and target
    X = df[selected_features]
    y = df["Diagnosis"].map({"B": 0, "M": 1})  # Encode diagnosis: B = 0, M = 1

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
