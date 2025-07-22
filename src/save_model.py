import joblib
import os

def save_model(model, filename="logistic_regression_model.pkl"):
    """
    Saves the trained model to disk using joblib.
    """
    # Create a 'models' directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Define the full path
    filepath = os.path.join("models", filename)

    # Save model
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
    

def save_metrics(metrics: dict, file_path="outputs/model_metrics.txt"):
    """
    Saves model evaluation metrics to a .txt file.

    Parameters:
    - metrics (dict): Dictionary with keys 'accuracy', 'precision', 'recall', 'f1_score'.
    - file_path (str): Path to save the metrics file.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        f.write("Model Evaluation Metrics\n")
        f.write("-------------------------\n")
        for metric_name, value in metrics.items():
            f.write(f"{metric_name.capitalize()}: {value:.3f}\n")

