import unittest
import pandas as pd
from src.load_data import load_breast_cancer_data

class TestLoadData(unittest.TestCase):
    def setUp(self):
        self.df = load_breast_cancer_data()

    def test_return_type(self):
        """Test that function returns a pandas DataFrame"""
        self.assertIsInstance(self.df, pd.DataFrame)

    def test_shape(self):
        """Test that the DataFrame has expected shape"""
        self.assertEqual(self.df.shape, (569, 31))  # 569 samples, 30 features + target

    def test_no_nulls(self):
        """Test that there are no missing values"""
        self.assertFalse(self.df.isnull().values.any())

if __name__ == '__main__':
    unittest.main()
