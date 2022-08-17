"""
Module for handling Bearing Dataset. Loads Bearing Dataset and creates evaluation
"""

from typing import Any, Iterable, List, Dict, Union

import pandas as pd
import mlflow

from fl_models.util.metrics import rmse, correlation_coefficient
from fl_models.util.dynamic_loader import load_model, load_dataset

from fl_models.abstract.abstract_dataset import AbstractDataset
from fl_models.abstract.abstract_model import AbstractModel

from fl_models.abstract.abstract_usecase import FederatedLearningUsecase


class BearingUseCase(FederatedLearningUsecase):
    """
    Class managing the model and dataset used in the bearing use case
    """

    def __init__(
        self,
        test_bearings: List[str] = None,
        log_mlflow: bool = True,
        **kwargs,
    ) -> None:
        """
        Creates a new bearing use case

        :param test_bearings: Bearings from the dataset that will be used for testing, defaults to \
            None
        :type test_bearings: List[str]
        :param log_mlflow: Whether the training should be logged to mlflow, defaults to True
        :type log_mlflow: bool, optional
        :raises ValueError: When test_bearings is missing
        """
        super().__init__(log_mlflow)

        if test_bearings is None:
            raise ValueError('Use Case is missing "test_bearings" parameter!')

        self.data: Dict[str, pd.DataFrame] = None
        self.labels: pd.Series = None
        self.abstract_model: AbstractModel = load_model(
            model_name=self.get_model_name(),
            learning_rate=kwargs.pop("learning_rate", None),
        )

        self.test_bearings = test_bearings
        self.dataset: AbstractDataset = load_dataset(
            self.get_dataset_name(),
            file_name=self.abstract_model.get_data_set_type().value,
            filter_list=self.test_bearings,
        )

        self.metrics = [rmse, correlation_coefficient]

        self.load_data()

        # Log model
        if log_mlflow:
            mlflow.log_param("test_bearings", test_bearings)

        # Warn in case there are unused arguments
        self.check_for_unused_args(kwargs)

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
            return self.data.drop(columns=["bearing_name"])

        grouped_data = self.data.groupby("bearing_name", as_index=False)
        return [
            grouped_data.get_group(bearing).drop("bearing_name", axis=1)
            for bearing in self.identifiers
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
            return self.labels["RUL"]

        grouped_labels = self.labels.groupby("bearing_name", as_index=False)
        return [
            grouped_labels.get_group(bearing)["RUL"] for bearing in self.identifiers
        ]

    def load_data(self):
        """
        Loads the bearing data used for training in memory further to train the model
        """

        full_df = self.dataset.get_dataframe()
        self.data = full_df.drop(["RUL"], axis=1)
        self.labels = full_df[["bearing_name", "RUL"]]
        self.identifiers = full_df["bearing_name"].unique()

    @staticmethod
    def get_model_name() -> str:
        """
        Getter for the model name

        :return: Model name
        :rtype: str
        """
        return "CNNSpectraFeatures"

    @staticmethod
    def get_dataset_name() -> str:
        """
        Getter for the dataset name

        :return: Dataset name
        :rtype: str
        """
        return "BearingDataset"


def load_class() -> BearingUseCase:
    """
    Getter for Dynamic Loader
    """
    return BearingUseCase
