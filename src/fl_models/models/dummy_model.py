"""
Module contains a dummy model
"""
from typing import List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import History

from fl_models.abstract.abstract_model import AbstractModel


class DummyModel(AbstractModel):
    """
    Implements a dummy model
    """

    def __init__(self, learning_rate: float = None):
        """
        Creates a Dummy Model

        :param learning_rate: Learning rate of the model, defaults to None
        :type learning_rate: float, optional
        """
        super().__init__(learning_rate=learning_rate)

        # pylint: disable=too-few-public-methods
        # Needed due to interface
        class DummyPredModel:
            """
            Dummy prediction model
            """

            # pylint: disable=unused-argument,no-method-argument
            # Has to conform to interface
            def summary(*args, **kwargs):
                """
                Prints the model summary
                """
                print("This is a Dummy!")

        self.prediction_model = DummyPredModel()

    # pylint: disable=unused-argument
    # Has to conform to interface
    @staticmethod
    def train(
        training_data: Union[pd.DataFrame, np.ndarray],
        training_labels: Union[pd.Series, np.ndarray],
        epochs: int,
        validation_data: Optional[
            Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]
        ] = None,
    ) -> History:
        """
        Does dummy training on the dataset.

        :param training_data: DataFrame or array of features. First axis is interpreted as number \
            of samples
        :type training_data: Union[pd.DataFrame, np.ndarray]
        :param training_labels: Series of labels. Has to be the same length as training_data
        :type training_labels: Union[pd.Series, np.ndarray]
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

        # pylint: disable=too-few-public-methods
        # Needed due to interface
        class DummyHistory:
            """
            Dummy history class
            """

            def __init__(self):
                """
                Creates a dummy training history object
                """
                self.history = {"loss": np.random.rand(2)}

        print("Did dummy training")
        return DummyHistory()

    @staticmethod
    def predict(data: Union[pd.DataFrame, np.ndarray]) -> pd.Series:
        """
        Takes a dataframe as input and performs a prediction on it.
        The first axis is interpreted as number of samples

        :param data: DataFrame or array that contains the features for prediction
        :type data: Union[pd.DataFrame, np.ndarray]
        :return: Predictions, has the length data.shape[0]
        :rtype: pd.Series
        """
        print("Did dummy predict")
        return pd.Series(
            np.random.rand(data.shape[0]),
            name="rul_predictions",
        )

    @staticmethod
    def get_weights() -> List[np.array]:
        """
        Dummy function, returns random weights

        :return: Random model weights
        :rtype: List[np.array]
        """
        return [np.random.rand(3) for _ in range(3)]

    # pylint: disable=unused-argument
    # Has to conform to interface
    @staticmethod
    def set_weights(new_weights: List[np.array]) -> None:
        """
        Dummy function, does nothing

        :param new_weights: New model weigths
        :type new_weights: List[np.array]
        """
        return


def load_class():
    """Returns the class in this file

    Returns:
        model_abc.AbstractModel
    """
    return DummyModel
