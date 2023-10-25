"""Import the important package"""
import unittest
import sys
import warnings
import io

from training.model_train import Trainer
from model.classifier import ANN
from dataset.load_dataset import DataLoader

warnings.filterwarnings("ignore")

sys.path.append("D:/IrisClassifier/IrisClassifier")


class UnitTesting(unittest.TestCase):
    """A class for unit testing the IrisClassifier project"""

    def setUp(self):
        """
        Setup test data and objects for testing

        - load the Iris Dataset
        - Preprocess the data and create train and test loaders
        - Initialise the ANN model and Trainer
        """
        self.dataset = DataLoader(dataFrame="D:/IrisClassifier/Iris.csv")
        self.train_loader, self.test_loader = self.dataset.preprocess_data()
        self.model = ANN()
        self.trainer = Trainer(
            epochs=500,
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.test_loader,
        )

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
        self.trainer.model_performance()
        self.trainer.model_evaluate(dataloader=self.test_loader, model=self.model)

        (
            accuracy,
            precision,
            recall,
            f1,
            _,
            _,
        ) = self.trainer.get_metrics(dataloader=self.test_loader, model=self.model)

        self.assertTrue(0.0 <= accuracy <= 1.0)
        self.assertTrue(0.0 <= precision <= 1.0)
        self.assertTrue(0.0 <= recall <= 1.0)
        self.assertTrue(0.0 <= f1 <= 1.0)

        expected_output = "Expected output message"

        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout

        print(expected_output)
        sys.stdout = old_stdout
        captured_output = new_stdout.getvalue().strip()
        captured_output = captured_output.strip()

        self.assertEqual(captured_output, expected_output)

    def test_evaluation_with_invalid(self):
        """
        Test the evaluation of the trained model with invalid data
        """
        self.test_loader = not None
        with self.assertRaises(Exception):
            self.trainer.model_evaluate(dataloader=self.test_loader, model=self.model)


if __name__ == "__main__":
    unittest.main()
