"""Import the libraries"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from evaluation.evaluate_metrics import ModelEvaluate


class ModelNotDefinedError(Exception):
    """Exception raised when the model is not defined."""

    def __init__(self, message="Model is not defined."):
        self.message = message
        super().__init__(self.message)


class Trainer:
    """
    A class for training a neural network model

    Args:
        epochs (init) : The number of training epochs (default is 100)
        model: The neural network model to be trained
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data

    Raises:
        RuntimeError: If model, train_loader, and val_loader are not provided

    Attributes:
        EPOCHS (int): Number of training Epochs
        history (dict): A dictionary to store training history
        model: The neural network model to be trained
        loss_function: The loss function is used for training
        optimizer: The optimizer for model parameters updates
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data

    """

    def __init__(self, epochs=100, model=None, train_loader=None, val_loader=None):
        if model is not None and train_loader is not None and val_loader is not None:
            self.total_epochs = epochs
            self.history = {
                "loss": [],
                "val_loss": [],
                "accuracy": [],
                "val_accuracy": [],
            }
            self.model = model
            self.loss_function = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=0.001)
            self.train_loader = train_loader
            self.val_loader = val_loader
        else:
            raise ModelNotDefinedError("Model is not defined.".title())

    def _model_prediction(self, model, data):
        """
        Compute predictions by taking the arg max of the predicted values

        Args:
            predicted: Predicted values from the model

        Returns:
            Tensor: Computed predictions
        """
        return model(data)

    def _compute_labels(self, predicted):
        """
        Compute labels by taking the arg max of the predicted values

        Args:
            predicted: Predicted values from the model

        Returns:
            Tensor: Computed labels
        """
        return torch.argmax(predicted, dim=1)

    def _compute_loss(self, predicted, actual):
        """
        Compute loss between predicted and actual values

        Args:
            predicted: Predicted values from the model
            Actual: Actual target values
        Returns:
            Tensor: Computed loss
        """
        return self.loss_function(predicted, actual)

    def _do_back_propagation(self, optimizer, loss_function):
        """
        Perform back propagation and update model parameters

        Args:
            optimizer: Model optimizer
            loss_function: Loss function

        Returns:
            None
        """
        optimizer.zero_grad()
        loss_function.backward()
        optimizer.step()

    def _display(
        self,
        epoch=None,
        total_train_loss=None,
        train_accuracy=None,
        total_val_loss=None,
        test_accuracy=None,
    ):
        """Display the model train and validation loss and accuracy"""
        print(f"Epoch {epoch}/{ self.total_epochs}")
        print(
            "[==========] loss - {} accuracy - {} - val_loss - {} val_accuracy - {} ".format(
                np.array(total_train_loss).mean(),
                train_accuracy,
                np.array(total_val_loss).mean(),
                test_accuracy,
            )
        )

    def train(self):
        """
        Train the neural networks

        Returns:
            None
        """
        total_train_loss = []
        for epoch in range(self.total_epochs):
            for x_batch, y_batch in self.train_loader:
                train_prediction = self._model_prediction(
                    model=self.model, data=x_batch
                )
                train_loss_compute = self._compute_loss(
                    predicted=train_prediction, actual=y_batch
                )
                self._do_back_propagation(
                    optimizer=self.optimizer, loss_function=train_loss_compute
                )

            train_predicted = self._compute_labels(train_prediction)
            total_train_loss.append(train_loss_compute.item())
            train_accuracy = accuracy_score(train_predicted, y_batch)
            print(train_accuracy)
            self.history["loss"].append(np.array(total_train_loss).mean())
            self.history["accuracy"].append(train_accuracy)

            total_val_loss = []

            for val_data, val_label in self.val_loader:
                test_prediction = self._model_prediction(
                    model=self.model, data=val_data
                )
                test_loss_compute = self._compute_loss(
                    predicted=test_prediction, actual=val_label
                )

            test_predicted = self._compute_labels(test_prediction)
            total_val_loss.append(test_loss_compute.item())
            test_accuracy = accuracy_score(test_predicted, val_label)
            print(test_accuracy)
            self.history["val_loss"].append(np.array(total_val_loss).mean())
            self.history["val_accuracy"].append(test_accuracy)

            self._display(
                total_train_loss=total_train_loss,
                epoch=epoch,
                train_accuracy=train_accuracy,
                total_val_loss=total_val_loss,
                test_accuracy=test_accuracy,
            )

    def model_performance(self):
        """Model performance will show"""
        _, axes = plt.subplots(1, 2)

        axes[0].plot(self.history["loss"], label="train_loss")
        axes[0].plot(self.history["val_loss"], label="val_loss")
        axes[0].set_title("train and test loss")

        axes[1].plot(self.history["accuracy"], label="train_accuracy")
        axes[1].plot(self.history["val_accuracy"], label="val_accuracy")
        axes[1].set_title("train and test accuracy")

        plt.show()

    def model_evaluate(self, dataloader=None, model=None):
        """
        Evaluate the machine learning models using data loaders and model

        Args:
            dataloader: A dataloader containing the evaluation data
            model: A NN model to be processed

        Returns:
            None

        Raises:
            ValueError: If either `dataloader` or `model` is not provided
        """
        evaluation = ModelEvaluate(dataloader=dataloader, model=model)
        evaluation._display_metrics_and_report()

    def get_metrics(self, dataloader=None, model=None):
        """
        Calculate and retrieve evaluation metrics for a machine learning model

        Args:
            dataloader: A Dataloader containing the evaluation data
            model: A machine learning model to be processed

        Returns:
            tuple: A tuple containing the following evaluation metrics and result
                - accuracy: The accuracy of model predictions
                - precision: The precision of model predictions
                - recall: The recall of model predictions
                - f1: The f1 score of model predictions
                - predict: The predict of model predictions
                - actual: The actual labels of features

        Raises:
            ValueError: If either `dataloader` or `model` is not provided

        Example usages:
            # Create instance of the class
            instance = Class()

            # Calculate metrics
            accuracy, precision, recall, f1, predict, actual = evaluation._evaluate()
        """
        evaluation = ModelEvaluate(dataloader=dataloader, model=model)
        (accuracy, precision, recall, f1, predict, actual) = evaluation._evaluate()
        return accuracy, precision, recall, f1, predict, actual


if __name__ == "__main__":
    pass
