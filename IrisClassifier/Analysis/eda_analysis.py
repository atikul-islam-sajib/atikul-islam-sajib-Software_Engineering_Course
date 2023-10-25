import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    """
    DataVisualizer is a class for visualizing data in a pandas DataFrame
        
    Parameters:
    ----------
    dataFrame: pandas DataFrame, optional
        The input DataFrame containing the data to be visualized
            
    Attributes:
    ----------
    dataFrame: pandas DataFrame
        The DataFrame containing this data to be visualized.
    """
    def __init__(self, dataFrame = None):
        if isinstance(dataFrame, pd.core.frame.DataFrame):
            self.dataFrame = dataFrame
            # self._show_pair_plot(dataset = self.dataFrame)
            # self._check_target_distribution(dataset = self.dataFrame)
        else:
            raise "Data Frame should be in the pandas format.".title()
    """
    _show_pair_plot() : Display the pair plot of the dataset using seaborn
    """
    def _show_pair_plot(self, dataset = None):
        if dataset is not None:
            sns.pairplot(dataset)
            plt.title("Pair plot of the Iris Classifier")
            plt.show()
        
        else:
            raise "Dataset should be passed.".title()
        
    """
    _check_target_distribution() : Display the target distribution of the dataset.
    """
    def _check_target_distribution(self, dataset = None):
        if dataset is not None:
            plt.title('Distribution of target class')
            dataset.iloc[:, -1].value_counts().plot(kind = 'barh')
            plt.xlabel('Quantity')
            plt.ylabel('Class')
            plt.show()
            
        else:
            raise "Dataset should be passed.".title()