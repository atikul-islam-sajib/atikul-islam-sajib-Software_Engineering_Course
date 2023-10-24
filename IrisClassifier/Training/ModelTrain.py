import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from Evaluation.evaluate_metrics import modeleEvaluate


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
            self.EPOCHS = epochs
            self.history = {'loss': [],
                            'val_loss': [],
                            'accuracy': [],
                            'val_accuracy': []
                            }
            self.model = model
            self.loss_function = nn.CrossEntropyLoss()
            self.optimizer = optim.AdamW(
                params=self.model.parameters(), lr=0.001)
            self.train_loader = train_loader
            self.val_loader = val_loader
        else:
            raise "Model is not defined.".title()
    """
    Perform forward pass and make predictions using the model
    
    Args:
        model: The neural network model
        data: Input data
    
    Returns:
        Tensor: Model Prediction
    """

    def _model_prediction(self, model, data):
        return model(data)

    """
    Compute labels by taking the argmax of the predicted values
    
    Args:
        predicted: Predicted values from the model
        
    Returns:
        Tensor: Computed labels
    """

    def _compute_labels(self, predicted):
        return torch.argmax(predicted, dim=1)
    """
    Compute loss between predicted and actual values
    
    Args:
        predicted: Predicted values from the model
        Actual: Actual target values
    Returns:
        Tensor: Computed loss
    """

    def _compute_loss(self, predicted, actual):
        return self.loss_function(predicted, actual)
    """
    Perform backpropagation and update model parameters
    
    Args:
        optimizer: Model optimizer
        loss_function: Loss function
    
    Returns:
        None
    """

    def _do_backpropagation(self, optimizer, loss_function):
        optimizer.zero_grad()
        loss_function.backward()
        optimizer.step()

    def _display(self, epoch=None, TRAIN_LOSS=None, train_accuracy=None, VAL_LOSS=None, test_accuracy=None):

        print("Epoch {}/{} ".format(epoch, self.EPOCHS))
        print("[==========] loss - {} accuracy - {} - val_loss - {} val_accuracy - {} ".format(np.array(TRAIN_LOSS).mean(),
                                                                                               train_accuracy,
                                                                                               np.array(VAL_LOSS).mean(),
                                                                                               test_accuracy))
    """
    Train the neural networks
    
    Returns:
        None
    """
    def train(self):
        TRAIN_LOSS = []
        for epoch in range(self.EPOCHS):
            for (X_batch, y_batch) in self.train_loader:
                train_prediction = self._model_prediction(
                    model=self.model, data=X_batch)
                train_loss_compute = self._compute_loss(
                    predicted=train_prediction, actual=y_batch)
                self._do_backpropagation(
                    optimizer=self.optimizer, loss_function=train_loss_compute)

            train_predicted = self._compute_labels(train_prediction)
            TRAIN_LOSS.append(train_loss_compute.item())
            train_accuracy = accuracy_score(train_predicted, y_batch)
            print(train_accuracy)
            self.history['loss'].append(np.array(TRAIN_LOSS).mean())
            self.history['accuracy'].append(train_accuracy)

            VAL_LOSS = []

            for (val_data, val_label) in self.val_loader:
                test_prediction = self._model_prediction(
                    model=self.model, data=val_data)
                test_loss_compute = self._compute_loss(
                    predicted=test_prediction, actual=val_label)

            test_predicted = self._compute_labels(test_prediction)
            VAL_LOSS.append(test_loss_compute.item())
            test_accuracy = accuracy_score(test_predicted, val_label)
            print(test_accuracy)
            self.history['val_loss'].append(np.array(VAL_LOSS).mean())
            self.history['val_accuracy'].append(test_accuracy)

            self._display(TRAIN_LOSS=TRAIN_LOSS,
                          epoch=epoch,
                          train_accuracy=train_accuracy,
                          VAL_LOSS=VAL_LOSS,
                          test_accuracy=test_accuracy)
    
    def model_performane(self):
        fig, axes = plt.subplots(1, 2)

        axes[0].plot(self.history['loss'], label = 'train_loss')
        axes[0].plot(self.history['val_loss'], label = 'val_loss')
        axes[0].set_title('train and test loss')

        axes[1].plot(self.history['accuracy'], label = 'train_accuracy')
        axes[1].plot(self.history['val_accuracy'], label = 'val_accuracy')
        axes[1].set_title('train and test accuracy')
        
        plt.show()
    
    def model_evaluate(self, dataloader = None, model = None):
        evaluation = modeleEvaluate(dataloader = dataloader, model = model)
        evaluation._display_metrics_and_report()
    
    def get_metrics(self, dataloader = None, model = None):
        evaluation = modeleEvaluate(dataloader = dataloader, model = model)
        accuracy, precision, recall, f1, predict, actual = evaluation._evaluate()
        return accuracy, precision, recall, f1, predict, actual 
            
if __name__ == "__main__":
    # trainer = Trainer(epochs = 20, model = model, train_loader = train_loader, val_loader = test_loader)
    # trainer.train()
    pass
