import pandas as pd
import os


def load_breast_cancer_data(file_path="C:\Proyectos_DATA_&_IA\Breast_Cancer_Project_DA-ML\Breast_cancer_project\data\wdbc.data"):
    """
    Load the Wisconsin Breast Cancer Diagnostic dataset.

    Parameters:
        file_path (str): Path to the .data file.

    Returns:
        pd.DataFrame: DataFrame with named columns and target encoded.
    """
    # Column names based on wdbc.names from UCI
    columns = [
        "ID", "Diagnosis",
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]

    # Load data
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")

    df = pd.read_csv(file_path, header=None, names=columns)

    # Drop ID column (not predictive)
    df.drop("ID", axis=1, inplace=True)

    # Encode diagnosis: M = 1 (Malignant), B = 0 (Benign)
    df["Diagnosis"] = df["Diagnosis"].map({"M": 1, "B": 0})

    return df
