import pandas as pd
import sys
import os

sys.path.append("D:/IrisClassifier/IrisClassifier")

from Analysis.EDA import DataVisualizer
from Preprocessing.featureEnginnering import preprocess

# This is for check
from Model.classifier import ANN
from Training.ModelTrain import Trainer

class loadDataset:
    """
    A class for loading and working with datasets in CSV format using Pandas

    Parameters:
    ----------
    dataFrame: str, optional
    The filename or path to CSV dataset to be loaded

    Raises:
    ------
    ValueError:
        If the provided dataframe does not have a '.csv' file extensions

    Attributes:
    ----------
    dataFrame : str
        The filename or path to be loaded in CSV dataset.
    """

    def __init__(self, dataFrame=None):
        self.dataFrame = None
        if dataFrame.split()[0][-3:] == "csv":
            self.dataFrame = dataFrame

            self.dataFrame = self._read_csv(dataset=self.dataFrame)

            # Intialise the object
            self.data_visualization = DataVisualizer(dataFrame=self.dataFrame)
            self.data_preprocessing = preprocess(dataFrame=self.dataFrame)

        else:
            raise "Data Frame should be in the CSV format".title()

    def _read_csv(self, dataset=None):
        dataset = pd.read_csv(dataset)
        # Check the dataset in pandas DataFrame or not
        if isinstance(dataset, pd.core.frame.DataFrame):
            self._show_dataset(dataset=dataset)

            return dataset
        else:
            raise "Data Frame should be in the pandas format".capitalize()

    """
    Display the first few rows of the provided dataset
    
    Parameters:
    ----------
    dataset : pandas DataFrame, optional
        The dataset to be displayed. If not provided the class attributes 'dataFrame' is used
    """

    def _show_dataset(self, dataset=None):
        print(dataset.head())

    """
    Show a visualization using the provided functions and plots data
    
    Parameters:
    ----------
    func : functions
        The visualization function to be executed
    
    plot : object
        The data to be used for plotting or visualization
    
    Returns:
    -------
        return : object
            The result of the visualization function
    """
    # Functions as parameters(83-103)
    def _show_visualization(self, func, plot):
        return func(plot)

    """
    Display the pair plot and distribution of target feature

    Returns:
    -------
    It will return two plots
    """

    def display(self):
        plot1 = self._show_visualization(
            self.data_visualization._show_pair_plot, self.dataFrame
        )
        plot2 = self._show_visualization(
            self.data_visualization._check_target_distribution, self.dataFrame
        )

        return plot1, plot2

    def _print_statement(self, X_train, X_test, y_train, y_test, train_loader):
        
        print("\n")
        print("X_train shape # {} ".format(X_train.shape), '\n')
        print("y_train shape # {} ".format(y_train.shape), '\n')
        print("X_test shape  # {} ".format(X_test.shape), '\n')
        print("y_test shape  # {} ".format(y_test.shape), '\n')

        train_data, train_label = next(iter(train_loader))

        print("Batch size # {} ".format(train_loader.batch_size), '\n')
        print("Train data shape with single batch_size  # {} ".format(
            train_data.shape), '\n')
        print("Train label shape with single batch_size # {} ".format(
            train_label.shape), '\n')

    def preprocess_data(self):
        """
        Drop the Id feature from the dataset
        Preprocess the data by encoding the target columns and scaling

        Returns:
        --------
        preprocessed_data : pandas DataFrame
            The preprocessed DataFrame with the encoded target columns and scaled

        """
        self.data_preprocessing._drop_column(dataset = self.dataFrame)
        
        new_dataset, scaling_dataset = self.data_preprocessing._do_encoding_and_scaling(
            dataset=self.dataFrame
        )
        self.dataFrame = scaling_dataset(new_dataset)
        
        
        # Display the dataset
        self._show_dataset(dataset=self.dataFrame)
        
        # Split the dataset train and test with respect to Torch
        X_train, X_test, y_train, y_test, TRAIN_LOADER, VAL_LOADER = self.data_preprocessing._train_test_split(
            dataset=self.dataFrame)
        
        # Display the train and test split result
        self._print_statement(X_train, X_test, y_train, y_test, TRAIN_LOADER)

        return TRAIN_LOADER, VAL_LOADER


if __name__ == "__main__":
    dataset = loadDataset(dataFrame="D:/IrisClassifier/Iris.csv")
    plot1, plot2 = dataset.display()
    train_loader, test_loader = dataset.preprocess_data()
    
    # This is for check
    model = ANN()
    model.total_trainable_parameters()
    import warnings

    # Ignore the specific warning about the deprecated softmax dimension choice
    warnings.filterwarnings("ignore")
    trainer = Trainer(epochs = 500, model = model, train_loader = train_loader, val_loader = test_loader)
    trainer.train()
    trainer.model_performane()
    trainer.model_evaluate(dataloader = test_loader, model = model)
