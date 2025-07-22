import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model using training data.
    
    Args:
        X_train (array-like): Features for training.
        y_train (array-like): Target labels for training.
    
    Returns:
        model: Trained Logistic Regression model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on test data and prints performance metrics.
    
    Args:
        model: Trained classification model.
        X_test (array-like): Features for testing.
        y_test (array-like): True target labels.
    """
    y_pred = model.predict(X_test)

    print("Model Evaluation Metrics")
    print("Accuracy: ", round(accuracy_score(y_test, y_pred), 3))
    print("Precision:", round(precision_score(y_test, y_pred), 3))
    print("Recall:   ", round(recall_score(y_test, y_pred), 3))
    print("F1-score: ", round(f1_score(y_test, y_pred), 3))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return y_pred


def plot_confusion_matrix(y_test, y_pred):
    """
    Plots the confusion matrix using seaborn heatmap.
    
    Args:
        y_test (array-like): True target labels.
        y_pred (array-like): Predicted labels.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train, random_state=42):
    """
    Trains a Random Forest Classifier.
    
    Parameters:
        X_train (DataFrame): Training features
        y_train (Series): Training labels
        random_state (int): Seed for reproducibility
        
    Returns:
        model (RandomForestClassifier): Trained model
    """
    model = RandomForestClassifier(random_state=random_state, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def tune_random_forest(X_train, y_train, cv=5):
    """
    Perform hyperparameter tuning using GridSearchCV for RandomForestClassifier.
    Returns the best estimator found.
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=cv, scoring='f1', n_jobs=-1, verbose=1)

    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    print("Best F1 score:", grid_search.best_score_)

    return grid_search.best_estimator_
