import unittest
from src.load_data import load_breast_cancer_data
from src.preprocessing import preprocess_data
from src.model_training import train_logistic_regression, evaluate_model
from sklearn.base import ClassifierMixin
import numpy as np

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        """Load and preprocess the dataset for testing."""
        df = load_breast_cancer_data()
        self.X_train, self.X_test, self.y_train, self.y_test = preprocess_data(df)

    def test_model_instance(self):
        """Check if the model returned is a scikit-learn classifier."""
        model = train_logistic_regression(self.X_train, self.y_train)
        self.assertIsInstance(model, ClassifierMixin)

    def test_prediction_shape(self):
        """Ensure model predictions have the same shape as y_test."""
        model = train_logistic_regression(self.X_train, self.y_train)
        y_pred = evaluate_model(model, self.X_test, self.y_test)
        self.assertEqual(len(y_pred), len(self.y_test))

if __name__ == '__main__':
    unittest.main()
