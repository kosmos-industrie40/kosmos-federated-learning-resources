"""Module that defines an example Federated Learning usecase"""

from typing import Any, Dict, Iterable, Union

import pandas as pd

from fl_models.abstract.abstract_model import AbstractModel
from fl_models.abstract.abstract_dataset import AbstractDataset
from fl_models.util.metrics import rmse

from fl_models.abstract.abstract_usecase import FederatedLearningUsecase
from fl_models.util.dynamic_loader import load_model, load_dataset


class MyUsecase(FederatedLearningUsecase):

    # pylint: disable=too-many-instance-attributes
    # All of these are necessary for the calculation
    def __init__(
        self,
        user_dict: str = None,
        target_column: str = None,
        identifier_column: str = None,
        log_mlflow: bool = True,
        learning_rate: float = None,
        **kwargs,
    ):
        """
        Constructor for the example usecase

        :param user_dict: Dictionary containing the dataframe data
        :type user_dict: str
        :param target_column: Name of the target column in the dictionary
        :type target_column: str
        :param identifier_column: Name of the column with the identifiers in the dictionary
        :type identifier_column: str
        :param log_mlflow: Selects whether training progress should be logged to mlflow, defaults
            to True
        :type log_mlflow: bool, optional
        :param learning_rate: Learning rate of the model, defaults to None
        :type learning_rate: float, optional
        """
        assert user_dict is not None, "Missing 'user_dict' setting in configuration."
        assert (
            target_column is not None
        ), "Missing 'target_column' setting in configuration."
        assert (
            identifier_column is not None
        ), "Missing 'identifier_column' setting in configuration."

        super().__init__(log_mlflow=log_mlflow)

        self.metrics = [rmse]

        # Initialize Dataset
        # -> parameters are the name of the dataset and the rest are the parameters of "MyDataset"
        self.dataset: AbstractDataset = load_dataset(
            self.get_dataset_name(), user_dict=user_dict
        )

        self.ident_col = identifier_column
        self.target_column = target_column

        # Load and split dataset
        full_df = self.dataset.get_dataframe()
        self.data: pd.DataFrame = full_df.drop(columns=[self.target_column])
        self.labels: pd.DataFrame = full_df[[self.ident_col, self.target_column]]
        self.identifiers: pd.Series = full_df[self.ident_col].unique()

        # Load model (needs input shape)
        # -> parameters are the name of the model and the rest are the parameters of "MyModel"
        self.abstract_model: AbstractModel = load_model(
            self.get_model_name(),
            learning_rate=learning_rate,
            input_shape=self.data.shape[1] - 1,  # -1 Since we still have the identifier column
        )

        # Warn in case there are unused arguments
        self.check_for_unused_args(kwargs)

    @staticmethod
    def get_model_name() -> str:
        """
        Getter for the model name

        :return: Model name
        :rtype: str
        """
        return "MyModel"

    @staticmethod
    def get_dataset_name() -> str:
        """
        Getter for the dataset name

        :return: Dataset name
        :rtype: str
        """
        return "MyDataset"

    def get_data(self, flat: bool = False) -> Iterable[Union[Any, pd.DataFrame]]:
        """
        Getter for usecase specific data

        :param flat: If False, the data will be grouped by identifier, if True the data is not
            grouped, defaults to False
        :type flat: bool, optional
        :return: Feature data for training/evaluation
        :rtype: Iterable[Union[Any, pd.DataFrame]]
        """
        if flat:
            return self.data.drop(columns=[self.ident_col])
        grouped_data = self.data.groupby(self.ident_col, as_index=False)
        return [
            grouped_data.get_group(val).drop(columns=[self.ident_col])
            for val in self.identifiers
        ]

    def get_labels(self, flat: bool = False) -> Iterable[Union[Any, pd.Series]]:
        """
        Getter for the usecase specific labels

        :param flat: If False, the labels will be grouped by identifier, defaults to False
        :type flat: bool, optional
        :return: Labels for training/evaluation
        :rtype: Iterable[Union[Any, pd.Series]]
        """
        if flat:
            return self.labels[self.target_column]
        grouped_labels = self.labels.groupby(self.ident_col, as_index=False)
        return [
            grouped_labels.get_group(val)[self.target_column]
            for val in self.identifiers
        ]


def load_class():
    return MyUsecase
