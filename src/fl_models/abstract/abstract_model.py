"""
Abstract interface for the usecase models
"""
import abc
from typing import Sequence, Callable, Optional, List, Tuple, Union

import pandas as pd
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.callbacks import History


class AbstractModel(abc.ABC):
    """
    Interface for Models
    """

    @abc.abstractmethod
    def __init__(self, learning_rate: float = None):
        """
        Base constructor of a model
        :param learning_rate: Learing rate of the model, defaults to None
        :type learning_rate: float, optional
        """
        self.prediction_model: Model = None
        self.trainings_history = None
        self.learning_rate = learning_rate

    @abc.abstractmethod
    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> pd.Series:
        """
        Takes a dataframe or an array as input and performs a prediction on it.
        The first axis is interpreted as number of samples

        :param data: DataFrame or array that contains the features for prediction
        :type data: Union[pd.DataFrame, np.ndarray]
        :return: Predictions, has the length data.shape[0]
        :rtype: pd.Series
        """
        raise NotImplementedError("'predict' has to be implemented by the subclass!")

    @abc.abstractmethod
    def train(
        self,
        training_data: Union[pd.DataFrame, np.ndarray],
        training_labels: pd.Series,
        epochs: int,
        validation_data: Optional[
            Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]
        ] = None,
    ) -> History:
        """
        Takes training data and labels and trains a model corresponding to the respective class.
        Also sets the prediction model.

        :param training_data: DataFrame or array of features. First axis is interpreted as number of
            samples
        :type training_data: Union[pd.DataFrame, np.ndarray]
        :param training_labels: Series of labels. Has to be the same length as training_data
        :type training_labels: pd.Series
        :param validation_data: Tuple containing (validation features, validation labels). \
            DataFrame or array of features used for validation. First axis is \
            interpreted as number of samples
        :type validation_data: Tuple[Union[pd.DataFrame, np.ndarray], \
            Union[pd.Series, np.ndarray]], optional
        :param epochs: Number of training epochs
        :type epochs: int
        :return: Returns the trainings history. Is expected to contain a list of losses under
            history.history["loss"]
        :rtype: History
        """
        raise NotImplementedError("'train' has to be implemented by the subclass!")

    def compute_metrics(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        labels: pd.Series,
        metrics_list: Sequence[Callable],
        logging_callback: Callable = lambda name, value: None,
    ) -> pd.DataFrame:
        """
        Evaluates the given dataset using the given metrics

        :param data: Feature data that is used for prediction
        :type data: Union[pd.DataFrame, np.ndarray]
        :param labels: Ground truth matching the data
        :type labels: pd.Series
        :param metrics_list: List of callables containing the metrics
        :type metrics_list: Sequence[Callable]
        :param logging_callback: Function used for logging to mlflow. Gets the metric name and
            current value as input, defaults to no logging
        :type logging_callback: Callable, optional
        :return: DataFrame with columns "metric" and "value"
        :rtype: pd.DataFrame
        """
        predictions = self.predict(data)
        result_dict = {}

        for index, metric in enumerate(metrics_list):

            metric_value = metric(labels, predictions)
            result_dict[index] = [metric.__name__, metric_value]

            # Call logging callback
            logging_callback(metric.__name__, metric_value)

        return pd.DataFrame.from_dict(
            result_dict, orient="index", columns=["metric", "value"]
        )

    def get_weights(self) -> List[np.array]:
        """
        Getter for model weights

        :return: List of model weights
        :rtype: List[np.array]
        """
        return self.prediction_model.get_weights()

    def set_weights(self, new_weights: List[np.array]) -> None:
        """
        Setter for model weights

        :param new_weights: New model weigths
        :type new_weights: List[np.array]
        """
        self.prediction_model.set_weights(weights=new_weights)
