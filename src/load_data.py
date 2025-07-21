import os
import pandas as pd

def load_breast_cancer_data():
    """
    Loads the Breast Cancer Wisconsin Diagnostic dataset from the local 'data' folder.
    Returns a pandas DataFrame.
    """
    # Construct robust relative path to 'data/wdbc.data'
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "..", "data", "wdbc.data")
    file_path = os.path.abspath(file_path)  # Optional: convert to absolute path

    # Define column names
    columns = [
        "ID", "Diagnosis",
        "Radius1", "Texture1", "Perimeter1", "Area1", "Smoothness1", "Compactness1", "Concavity1", "ConcavePoints1", "Symmetry1", "FractalDimension1",
        "Radius2", "Texture2", "Perimeter2", "Area2", "Smoothness2", "Compactness2", "Concavity2", "ConcavePoints2", "Symmetry2", "FractalDimension2",
        "Radius3", "Texture3", "Perimeter3", "Area3", "Smoothness3", "Compactness3", "Concavity3", "ConcavePoints3", "Symmetry3", "FractalDimension3"
    ]

    # Load data
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")

    df = pd.read_csv(file_path, header=None, names=columns)

    # Drop ID column (not predictive)
    df.drop("ID", axis=1, inplace=True)

    return df
