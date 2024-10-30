import unittest
import numpy as np
import pandas as pd
import os
import logging
from sklearn.ensemble import RandomForestClassifier

from train_model import load_data, process_data, train_model
from train_model import evaluate_model, save_model

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class TestMLFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up test environment, including loading sample data."""
        cls.data_filepath = "../data/census.csv" 
        logging.info("Loading data from %s", cls.data_filepath)
        cls.data = load_data(cls.data_filepath)
        cls.cat_features = [
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"
        ]
        cls.label = "salary"
        
        # Define paths for the model, encoder, and label binarizer
        cls.model_filepath = "../model/model.joblib"
        cls.encoder_filepath = "../model/encoder.joblib"
        cls.lb_filepath = "../model/label_binarizer.joblib"

    def test_load_data(self):
        """Test if load_data returns a DataFrame."""
        logging.info("Testing load_data function")
        data = load_data(self.data_filepath)
        self.assertIsInstance(data, pd.DataFrame)
        logging.info("load_data returned a DataFrame")

    def test_process_data(self):
        """Test if process_data returns the expected types."""
        logging.info("Testing process_data function")
        sample_data = self.data.sample(100, random_state=42)
        X, y, encoder, lb = process_data(
            sample_data, categorical_features=self.cat_features, 
            label=self.label, training=True)
        
        # Check if X is a numpy array
        self.assertIsInstance(X, np.ndarray)
        logging.info("process_data returned X as ndarray")
        
        # Check if y is a numpy array
        self.assertIsInstance(y, np.ndarray)
        logging.info("process_data returned y as ndarray")
        
        # Check if encoder and lb are not None
        self.assertIsNotNone(encoder)
        self.assertIsNotNone(lb)
        logging.info("process_data returned encoder and lb")

    def test_train_model(self):
        """Test if train_model returns a model instance."""
        logging.info("Testing train_model function")
        sample_data = self.data.sample(100, random_state=42)
        X, y, _, _ = process_data(
            sample_data, categorical_features=self.cat_features, label=self.label, training=True
        )
        model = train_model(X, y)
        
        # Check if model is an instance of RandomForestClassifier
        self.assertIsInstance(model, RandomForestClassifier)
        logging.info("train_model returned a RandomForestClassifier model instance")

    def test_evaluate_model(self):
        """Test if evaluate_model returns the expected metric types."""
        logging.info("Testing evaluate_model function")
        sample_data = self.data.sample(100, random_state=42)
        X, y, _, _ = process_data(
            sample_data, categorical_features=self.cat_features, label=self.label, training=True
        )
        model = train_model(X, y)
        
        # Evaluate the model
        accuracy, precision, recall, f1 = evaluate_model(model, X, y)
        
        # Check if each metric is a float
        self.assertIsInstance(accuracy, float)
        self.assertIsInstance(precision, float)
        self.assertIsInstance(recall, float)
        self.assertIsInstance(f1, float)
        logging.info("evaluate_model returned accuracy: %f, precision: %f, recall: %f, f1: %f", accuracy, precision, recall, f1)

    def test_save_model(self):
        """Test if the model, encoder, and label binarizer are saved as files."""
        logging.info("Testing save_model function")
        sample_data = self.data.sample(100, random_state=42)
        X, y, encoder, lb = process_data(
            sample_data, categorical_features=self.cat_features, label=self.label, training=True
        )
        model = train_model(X, y)
        
        # Save the model, encoder, and label binarizer
        save_model(model, encoder, lb, self.model_filepath, self.encoder_filepath, self.lb_filepath)
        
        # Check if the files were created
        self.assertTrue(os.path.isfile(self.model_filepath), "Model file was not saved")
        logging.info("Model file saved: %s", self.model_filepath)
        
        self.assertTrue(os.path.isfile(self.encoder_filepath), "Encoder file was not saved")
        logging.info("Encoder file saved: %s", self.encoder_filepath)
        
        self.assertTrue(os.path.isfile(self.lb_filepath), "Label binarizer file was not saved")
        logging.info("Label binarizer file saved: %s", self.lb_filepath)
    '''    
    @classmethod
    def tearDownClass(cls):
        """Clean up any files created during the tests."""
        logging.info("Cleaning up test files")
        for filepath in [cls.model_filepath, cls.encoder_filepath, cls.lb_filepath]:
            if os.path.isfile(filepath):
                os.remove(filepath)
                logging.info("Deleted file: %s", filepath)
    '''
if __name__ == "__main__":
    unittest.main()
