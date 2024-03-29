"""Import all important libraries"""
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


class ModelEvaluate:
    """
    A class for evaluating Deep Learning models using various metrics

    Args:
        dataloader : dataloader for evaluation
        model : The NN model to evaluate

    Raises:
        TypeError: If the provided dataloader is not of type DataLoader
    """

    def __init__(self, dataloader=None, model=None):
        if isinstance(dataloader, torch.utils.data.dataloader.DataLoader):
            self.dataloader = dataloader
            self.model = model
            self.accuracy = None
            self.precision = None
            self.recall = None
            self.f1 = None
        else:
            raise "dataloader should be in the torch dataloader format".title()

    """
    Computes the number of correct predictions
        
    Args:
        predict (list): List of the predicted labels
        actual (list): List of true labels
        
    Returns:
        int: Number of correct predictions
        
    Raises:
        TypeError: If the predict and actual are not in list
    """
    # anonymous functions based on functional programming

    def _compute_correct_prediction(self, predict=None, actual=None):
        if isinstance(predict, list) and isinstance(actual, list):
            correct_predict = sum(map(lambda x, y: x == y, predict, actual))
            return correct_predict
        else:
            raise "pass the predict & actual value in correct format".title()

    """
    Computes the number of correct predictions
        
    Args:
        predict (list): List of the predicted labels
        actual (list): List of true labels
        
    Returns:
        int: Number of correct predictions
        
    Raises:
        TypeError: If the predict and actual are not in list
    """
    # Only final data structures

    def _evaluate_accuracy(self, predict=None, actual=None):
        correct = sum(p == a for p, a in zip(predict, actual))
        accuracy = correct/len(actual)
        return accuracy

    """
    Computes the number of correct predictions
        
    Args:
        predict (list): List of the predicted labels
        actual (list): List of true labels
        
    Returns:
        int: Number of correct predictions
        
    Raises:
        TypeError: If the predict and actual are not in list
    """

    def _evaluate_precision(self, predict=None, actual=None):
        return precision_score(actual, predict, average="macro")

    """
    Computes the number of correct predictions
        
    Args:
        predict (list): List of the predicted labels
        actual (list): List of true labels
        
    Returns:
        int: Number of correct predictions
        
    Raises:
        TypeError: If the predict and actual are not in list
    """

    def _evaluate_recall(self, predict=None, actual=None):
        return recall_score(actual, predict, average="macro")

    """
    Computes the number of correct predictions
        
    Args:
        predict (list): List of the predicted labels
        actual (list): List of true labels
        
    Returns:
        int: Number of correct predictions
        
    Raises:
        TypeError: If the predict and actual are not in list
    """

    def _evaluate_f1(self, predict=None, actual=None):
        return f1_score(actual, predict, average="macro")

    """
    Computes the number of correct predictions
        
    Args:
        predict (list): List of the predicted labels
        actual (list): List of true labels
        
    Returns:
        int: Number of correct predictions
        
    Raises:
        TypeError: If the predict and actual are not in list
    """

    def _evaluation_metrics(self, predict=None, actual=None):
        if predict is not None and actual is not None:
            accuracy = self._evaluate_accuracy(predict=predict, actual=actual)
            precision = self._evaluate_precision(predict=predict, actual=actual)
            recall = self._evaluate_recall(predict=predict, actual=actual)
            f1 = self._evaluate_f1(predict=predict, actual=actual)

            return accuracy, precision, recall, f1
        else:
            raise "pass the predict & actual value in correct format".title()

    """
    Displays evaluation metrics
    
    Args:
        metrics (zip): A zip object containing evaluation metrics
        
    Raises:
        TypeError: If the metrics is not a zip object.
    """

    def _display(self, metrics=None):
        if isinstance(metrics, zip):
            print("\nEvaluation Metrics is given below.\n")
            for acc, pre, re, f1 in metrics:
                print(f"accuracy  # {acc} ".upper())
                print(f"precision # {pre} ".upper())
                print(f"recall    # {re} ".upper())
                print(f"f1 score  # {f1} ".upper())
        else:
            raise "metrics should be in zip format".title()

    """
    Computes the number of correct predictions
        
    Args:
        predict (list): List of the predicted labels
        actual (list): List of true labels
        
    """

    def _classification_report_show(self, predict=None, actual=None):
        print(classification_report(actual, predict))

    """
    Evaluates the NN model using provided data loader 
    """

    def _evaluate(self):
        if isinstance(self.dataloader, torch.utils.data.dataloader.DataLoader):
            predict = []
            actual = []
            compute_correct_predict = []
            for X_batch, y_batch in self.dataloader:
                predicted = self.model(X_batch)
                predicted = torch.argmax(predicted, dim=1)
                predict.extend(predicted)
                actual.extend(y_batch)
                compute_correct_predict.append(
                    self._compute_correct_prediction(
                        predict=predict, actual=actual
                    ).numpy()
                )

            accuracy, precision, recall, f1 = self._evaluation_metrics(
                predict=predict, actual=actual
            )

            return accuracy, precision, recall, f1, predict, actual

        else:
            raise "data loader should be in the torch form"

    def _display_metrics_and_report(self):
        accuracy, precision, recall, f1, predict, actual = self._evaluate()
        self._display(metrics=zip([accuracy], [precision], [recall], [f1]))

        print("\nThe classification report is given below.\n")
        self._classification_report_show(predict=predict, actual=actual)


if __name__ == "__main__":
    # model evaluate(dataloader = dataloader)
    pass
