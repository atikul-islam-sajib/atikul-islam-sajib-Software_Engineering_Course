import numpy
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class preprocess:
    """
    preprocess is a class for encoing and scaling a pandas DataFrame

    Parameters:
    ---------
    dataFrame : pandas DataFrame, optional
        This input DataFrame to be processed

    Attributes:
    ----------
    dataFrame : pandas DataFrame
        The DataFrame containing the data to be processed

    new_dataset : pandas DataFrame
        The processed DataFrame after encong and scaling
    """

    def __init__(self, dataFrame=None):
        if isinstance(dataFrame, pd.core.frame.DataFrame):
            self.datFrame = dataFrame
            # self.new_dataset = self._do_encoding(dataset = self.datFrame)
        else:
            raise "Data Frame should be in the pandas dataframe".title()

    """Drop the Id column from the dataset"""

    def _drop_column(self, dataset=None):
        dataset.drop(['Id'], axis=1, inplace = True)

    # Closures + Function as return values
    def _do_encoding_and_scaling(self, dataset=None):
        if dataset is not None:
            # Do the encoding for the target feature
            dataset.iloc[:, -1] = dataset.iloc[:, -1].\
                map({species: index for index, species in enumerate(
                    dataset.iloc[:, -1].unique())})
            # Create a function named sacling that will return sacling of the Independent Features

            def scaling(df):
                scaler = StandardScaler()
                transform_df = scaler.fit_transform(df.iloc[:, :-1])
                independent = pd.DataFrame(transform_df)
                dependent = pd.DataFrame(df.iloc[:, -1])
                new_df = pd.concat([independent, dependent], axis=1)

                return new_df

            return dataset, scaling
        else:
            raise "dataset is empty while calling ecoding".title()
    """
        Specific traget class extraction
        
        Parameters:
        -----------
            X : list or array like
                The independent features
            y : list or array like
                The dependent features
            
        Returns:
        --------
        filter_X : list
            It determines the independent features based on filter
        filter_y : list
            It determines the dependent features based on fileter
    """
    # Gigher order functions

    def _specific_target_class(self, X, y, func):
        # Take the target columns
        target_class = self.dataFrame.iloc[:, -1].unique()
        # to filter the pairs
        filter_data = filter(func(target_class), zip(X, y))
        filter_X, filter_y = zip(*filter_data)

        return filter_X, filter_y

    def _filter_target_class(self, target_class):
        # Return the filter by target class
        return lambda data: data[1] in target_class

    def _train_test_split_filter(self, dataset=None):
        # Split the data into independent & dependent
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        # If we want filter based target class
        filter_X, filter_y = self._filter_target_class(
            X, y, self._filter_target_class)

        return filter_X, filter_y
    """
    Split the dataset into train and test
    
    Parameters:
    -----------
    dataset : Pandas DataFrame, optional
        The dataset to be splitted. If not provided then the class attribute dataFrame is used
    
    Returns:
    --------
        X_train : The feature of training set
        y_train : The feature of training set
        X_test  : The feature of testing set
        y_test  : The feature of testing set
    """

    def _train_test_split(self, dataset=None):
        try:
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42)

            TRAIN_LOADER, VAL_LOADER = self._data_loader(
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
            )
            return X_train, X_test, y_train, y_test, TRAIN_LOADER, VAL_LOADER

        except Exception as e:
            print("The exception is caught : {} ".format(e)).title()

    def _data_loader(self, X_train=None, X_test=None, y_train=None, y_test=None):
        if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
            X_train = torch.tensor(
                data=X_train, dtype=torch.float32)
            X_test = torch.tensor(
                data=X_test, dtype=torch.float32)

            TRAIN_LOADER = DataLoader(
                dataset=list(zip(X_train, y_train)), batch_size=16, shuffle=True)
            VAL_LOADER = DataLoader(
                dataset=list(zip(X_test, y_test)), batch_size=16, shuffle=True)

            return TRAIN_LOADER, VAL_LOADER

        else:
            raise "Splitting is not done succeessfully".title()

# if __name__ == "__main__":
#     pass
