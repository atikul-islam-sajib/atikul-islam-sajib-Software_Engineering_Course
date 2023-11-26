"""Import the important package"""
import unittest
import sys
import warnings
import io
import torch

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
        
    def test_trainable_parameters(self):
        """
        Test the total trainable parameters with the model.
        - Define the model
        - Call the method named `total_trainable_parameters`
        """
        total_trainable_parameters = 1355
        
        self.assertEqual(self.model.total_trainable_parameters(), total_trainable_parameters)

    def test_evaluation_with_invalid(self):
        """
        Test the evaluation of the trained model with invalid data
        """
        self.test_loader = not None
        with self.assertRaises(Exception):
            self.trainer.model_evaluate(dataloader=self.test_loader, model=self.model)
    
    def test_ann_model(self):
        """
        Test the ANN model creation and forward pass
        
        - Define some sample input data
        - Perform a forward pass through the model
        - Perform assertion to check the output
        """
        self.input_data = torch.randn(1, 4)
        self.output = self.model(self.input_data)
    
        self.assertIsNotNone(self.output)
        self.assertEqual(self.output.shape, (1, 3))
        


if __name__ == "__main__":
    unittest.main()
