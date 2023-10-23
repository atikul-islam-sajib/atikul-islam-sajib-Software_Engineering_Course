import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

class modeleEvaluate:
    def __init__(self, dataloader = None, model = None):
        if isinstance(dataloader, torch.utils.data.dataloader.DataLoader):
            self.dataloader = dataloader
            self.model = model
        else:
            raise "dataloader should be in the torch dataloader format".title()
        
    # anonymous functions based on functional programming
    def _compute_correct_prediction(self, predict = None, actual = None):
        if isinstance(predict, list) and isinstance(actual, list):
            correct_predict = sum(map(lambda x, y: x == y, predict, actual))
            return correct_predict
        else:
            raise "pass the predict & actual value in correct format".title()
        
    # Only final data structures
    def _evaluate_accuracy(self, predict = None, actual = None):
        return accuracy_score(actual, predict)

    def _evaluate_precision(self, predict = None, actual = None):
        return precision_score(actual, predict, average = 'macro')

    def _evaluate_recall(self, predict = None, actual = None):
        return recall_score(actual, predict, average = 'macro')

    def _evaluate_f1(self, predict = None, actual = None):
        return f1_score(actual, predict, average = 'macro')

    def _evaluation_metrics(self, predict = None, actual = None):
        if predict is not None and actual is not None:
            accuracy = self._evaluate_accuracy(predict = predict, actual = actual)
            precision = self._evaluate_precision(predict = predict, actual = actual)
            recall  = self._evaluate_recall(predict = predict, actual = actual)
            f1 = self._evaluate_f1(predict = predict, actual = actual)

            return accuracy, precision, recall, f1
        else:
            raise "pass the predict & actual value in correct format".title()

    def _display(self, metrics = None):
        if isinstance(metrics, zip):
            print("\nEvaluation Metrics is given below.\n")
            for acc, pre, re, f1 in metrics:
                print("accuracy  # {} ".format(acc).upper())
                print("precision # {} ".format(pre).upper())
                print("recall    # {} ".format(re).upper())
                print("f1 score  # {} ".format(f1).upper())
        else:
            raise "metrics should be in zip format".title()

    def _classification_report_show(self, predict = None, actual = None):
        print(classification_report(actual, predict))

    def _evaluate(self):
        if isinstance(self.dataloader, torch.utils.data.dataloader.DataLoader):
            predict = []
            actual  = []
            compute_correct_predict = []
            for (X_batch, y_batch) in self.dataloader:
                predicted = self.model(X_batch)
                predicted = torch.argmax(predicted, dim = 1)
                predict.extend(predicted)
                actual.extend(y_batch)
                compute_correct_predict.append(self._compute_correct_prediction(predict = predict, actual = actual).numpy())
            
            accuracy, precision, recall, f1 = self._evaluation_metrics(predict = predict, actual = actual)
            self._display(metrics = zip([accuracy], [precision], [recall], [f1]))
            
            print("\nThe classification report is given below.\n")
            self._classification_report_show(predict = predict, actual = actual)

        else:
            raise "data loader should be in the torch form"
        
if __name__ == "__main__":
    # modeleEvaluate(dataloader = dataloader)
    pass
    