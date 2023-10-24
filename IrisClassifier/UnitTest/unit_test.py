import unittest
import sys
import warnings
import pytest
import io

warnings.filterwarnings("ignore")

sys.path.append("D:/IrisClassifier/IrisClassifier")

from Analysis.EDA import DataVisualizer
from Preprocessing.featureEnginnering import preprocess
from Dataset.loadDataset import loadDataset
from Model.classifier import ANN
from Training.ModelTrain import Trainer

class UnitTesting(unittest.TestCase):
    """A class for unit testing the IrisClassifier project"""

    def setUp(self):
        """
        Setup test data and objects for testing

        - load the Iris Dataset
        - Preprocess the data and create train and test loaders
        - Initialise the ANN model and Trainer
        """
        self.dataset = loadDataset(dataFrame="D:/IrisClassifier/Iris.csv")
        self.train_loader, self.test_loader = self.dataset.preprocess_data()
        self.model = ANN()
        self.trainer = Trainer(epochs=500, model=self.model,
                               train_loader=self.train_loader, val_loader=self.test_loader)

    def test_evaluation_with_valid(self):
        """
        Test the evaluation of the trained model with validation data

        - Train the model
        - Check model performance
        - Evaluate the model
        - Check the accuracy, precision, recall, f1 score
        - Check the captured output against as expected message
        """
        self.trainer.train()
        self.trainer.model_performane()
        self.trainer.model_evaluate(
            dataloader=self.test_loader, model=self.model)

        self.accuracy, self.precision, self.recall, self.f1, self.predict, self.actual = self.trainer.get_metrics(
            dataloader=self.test_loader, model=self.model)

        self.assertTrue(
            self.accuracy >= 0.0 and self.accuracy <= 1.0)
        self.assertTrue(
            self.precision >= 0.0 and self.precision <= 1.0)
        self.assertTrue(
            self.recall >= 0.0 and self.recall <= 1.0)
        self.assertTrue(
            self.f1 >= 0.0 and self.f1 <= 1.0)

        expected_output = "Expected output message"

        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout

        print(expected_output)
        sys.stdout = old_stdout
        captured_output = new_stdout.getvalue().strip()
        captured_output = captured_output.strip()

        self.assertEqual(captured_output, expected_output)


if __name__ == "__main__":
    unittest.main()
