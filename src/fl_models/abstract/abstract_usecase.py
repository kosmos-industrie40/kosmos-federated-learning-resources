"""Module that defines abstract class for Federated Learning usecase"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Tuple, Callable, Union
from warnings import warn

import numpy as np
import pandas as pd
import mlflow
from flwr.common import Weights
from tqdm import tqdm

from fl_models.abstract.abstract_model import AbstractModel
from fl_models.abstract.abstract_dataset import AbstractDataset


class FederatedLearningUsecase(ABC):
    """
    Abstract class for Federated Learning usecase. All subclasses should call the `__init__` \
    function

    Subclasses have to overwrite the following functions:
    * `get_model_name`
    * `get_dataset_name`

    Furthermore, subclasses might have to overwrite the following functions:
    * `get_data`
    * `get_labels`
    * `get_identifiers`
    * `get_number_of_samples`

    Very specific usecases might also have to overwrite:
    * `eval_fn`
    """

    # pylint: disable=too-many-instance-attributes
    # All of these are necessary for the calculation
    def __init__(self, log_mlflow: bool = True):
        """
        Base constructor for usecases

        :param log_mlflow: Selects whether training progress should be logged to mlflow, defaults
            to True
        :type log_mlflow: bool, optional
        """
        self.current_fed_rnd = 0

        self.metrics = []
        self.abstract_model: AbstractModel = None
        self.dataset: AbstractDataset = None
        self.log_mlflow = log_mlflow

        self.data: Union[pd.DataFrame, np.ndarray] = None
        self.labels: Union[pd.DataFrame, np.ndarray] = None
        self.identifiers: Union[pd.Series, np.ndarray] = None

    ### ABSTRACT AND USECASE SPECIFIC METHODS ###
    ### Subclasses might have/need to overwrite these methods

    @staticmethod
    @abstractmethod
    def get_model_name() -> str:
        """
        Getter for the model name

        :return: Model name
        :rtype: str
        """
        raise NotImplementedError("Subclass needs to overwrite get_model_name()")

    @staticmethod
    @abstractmethod
    def get_dataset_name() -> str:
        """
        Getter for the dataset name

        :return: Dataset name
        :rtype: str
        """
        raise NotImplementedError("Subclass needs to overwrite get_dataset_name()")

    def get_data(
        self, flat: bool = False
    ) -> Iterable[Union[Any, np.ndarray, pd.DataFrame]]:
        """
        Getter for usecase specific data

        :param flat: If False, the data will be grouped by identifier, if True the data is not
            grouped, defaults to False
        :type flat: bool, optional
        :return: Feature data for training/evaluation
        :rtype: Iterable[Union[Any, np.ndarray, pd.DataFrame]]
        """
        warn(f"This implementation ignores the {flat} parameter!")
        return self.data

    def get_labels(
        self, flat: bool = False
    ) -> Iterable[Union[Any, np.ndarray, pd.Series]]:
        """
        Getter for the usecase specific labels

        :param flat: If False, the labels will be grouped by identifier, defaults to False
        :type flat: bool, optional
        :return: Labels for training/evaluation
        :rtype: Iterable[Union[Any, np.ndarray, pd.Series]]
        """
        warn(f"This implementation ignores the {flat} parameter!")
        return self.labels

    def get_identifiers(self) -> Union[np.ndarray, pd.Series]:
        """
        Getter for the identifiers of the data/label groups

        :return: Array or Series containing the identifiers
        :rtype: Union[np.ndarray, pd.Series]
        """
        return self.identifiers

    def get_number_of_samples(self) -> int:
        """
        Getter for the number of samples in the dataset

        :return: Number of samples
        :rtype: int
        """
        return len(self.data)

    ### METHODS RELATED TO FEDERATED EVALUATION ###

    def eval_fn(self, weights: Weights) -> Tuple[float, Dict[str, float]]:
        """
        Defining the model evaluation function. Overwrite for more specific use cases.
        Called by flwr.

        :param weights: Model weights resulting from federated averaging
        :type weights: Weights
        :return: Tuple of RMSE, Dict[metric_name, metric_value]
        :rtype: Tuple[float, Dict[str, float]]
        """
        self.abstract_model.set_weights(weights)

        # Calculate metrics per identifier
        metrics_list = []
        identifiers = self.get_identifiers()
        for current_ident, current_data, current_label in tqdm(
            zip(identifiers, self.get_data(), self.get_labels()),
            total=len(identifiers),
            desc="Evaluating dataset...",
        ):
            current_metric = self.abstract_model.compute_metrics(
                data=current_data,
                labels=current_label,
                metrics_list=self.metrics,
                logging_callback=self.create_metric_callback(str(current_ident)),
            )

            metrics_list.append(current_metric)

        mean_df = pd.concat(metrics_list).groupby("metric", as_index=False).mean()
        return_val = self.create_dictionary_from_metric_df(mean_df, "RMSE")
        self.current_fed_rnd += 1
        return return_val

    def create_metric_callback(self, identifier: str) -> Callable:
        """
        Creates a callback function that takes the name of a metric and its value as input and logs
        them under <metric_name>_<identifier> in mlflow

        :param identifier String that will be appended to the metric name when logging

        :return: Function with (metric_name, metric_value) as parameters and logs them to mlflow
        """
        if not self.log_mlflow:
            return lambda name, value: None

        def callback(metric_name: str, metric_value):
            mlflow.log_metric(
                f"{metric_name}_{identifier}",
                metric_value,
                self.current_fed_rnd,
            )

        return callback

    def create_dictionary_from_metric_df(
        self, mean_metric_df: pd.DataFrame, loss_function_key: str = "RMSE"
    ) -> Tuple[float, Dict[str, float]]:
        """
        Converts the mean metrics to a dictionary and logs to mlflow

        :param mean_metric_df: DataFrame containing the mean values for each metric. Has the columns
            "metric" and "value"
        :type mean_metric_df: pd.DataFrame
        :param loss_function_key: The name of the metric that is used as loss function, defaults to
            "RMSE"
        :type loss_function_key: str, optional
        :return: A Tuple [loss value, Dict[metric name -> metric value]]
        :rtype: Tuple[float, Dict[str, float]]
        """
        loss = None
        avg_dict = {}
        for _, row in mean_metric_df.iterrows():
            c_metric = row["metric"]
            c_value = row["value"]

            if self.log_mlflow:
                mlflow.log_metric(c_metric, c_value, self.current_fed_rnd)

            if c_metric == loss_function_key:
                loss = c_value

            avg_dict[c_metric] = c_value

        return (
            loss,
            avg_dict,
        )

    ### DEFAULT GETTERS ###

    def get_model(self) -> AbstractModel:
        """
        Getter for the model

        :return: The usecase model
        :rtype: AbstractModel
        """
        return self.abstract_model

    def get_dataset(self) -> AbstractDataset:
        """
        Getter for the usecase dataset

        :return: The dataset of the usecase
        :rtype: AbstractDataset
        """
        return self.dataset

    ### UTILITY ###
    @staticmethod
    def check_for_unused_args(kwargs: Dict[str, Any]):
        """
        Iterates through the given dictionary and warns for each key that it contains

        :param kwargs: Dictionary containing the remaining keyword arguments from the usecase \
            instantiation
        :type kwargs: Dict[str, Any]
        """
        if len(kwargs) > 0:
            warn_string = ""
            for key, item in kwargs.items():
                warn_string += f" {key} = {item},"
            warn_string = warn_string.strip()[:-1]
            warn(f"Got the following unexpected parameters: {warn_string}")
