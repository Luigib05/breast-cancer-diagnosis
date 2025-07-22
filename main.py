from src.load_data import load_breast_cancer_data

df = load_breast_cancer_data()

print(df.head())
print(df.shape)
print(df.info())

from src.preprocessing import preprocess_data
X_train, X_test, y_train, y_test = preprocess_data(df)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

from src.model_training import train_logistic_regression, evaluate_model, plot_confusion_matrix

# Train the model
model = train_logistic_regression(X_train, y_train)

# Evaluate the model and get predictions
y_pred = evaluate_model(model, X_test, y_test)

# Plot confusion matrix
#plot_confusion_matrix(y_test, y_pred)


from src.model_training import train_random_forest, evaluate_model, plot_confusion_matrix

# Train model
rf_model = train_random_forest(X_train, y_train)

# Evaluate
rf_predictions = evaluate_model(rf_model, X_test, y_test)

# Plot confusion matrix
#plot_confusion_matrix(y_test, rf_predictions)

from src.model_training import tune_random_forest, evaluate_model, plot_confusion_matrix

# Train with tuned hyperparameters
best_rf_model = tune_random_forest(X_train, y_train)

# Evaluate
print("\nTuned Random Forest Results:")
best_rf_predictions = evaluate_model(best_rf_model, X_test, y_test)
#plot_confusion_matrix(y_test, best_rf_predictions)


#from src.save_model import save_model

# Save the trained logistic regression model
#save_model(model)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.save_model import save_metrics

# Predict test set
y_pred = model.predict(X_test)

# Compute metrics
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred)
}

# Save metrics to file
save_metrics(metrics)

from joblib import dump
dump(model, "model.joblib")

