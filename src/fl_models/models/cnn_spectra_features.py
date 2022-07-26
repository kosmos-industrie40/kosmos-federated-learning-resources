"""
Model predicting remaining useful lifetime using a CNN on spectra
"""
from typing import Optional, Tuple, Union
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import History

from fl_models.cnn_model import fit_cnn, build_cnn, reformat_spectra
from fl_models.abstract.abstract_model import AbstractModel
from fl_models.data_set_type import DataSetType


class CNNSpectraFeatures(AbstractModel):
    """
    Manages a model that computes the rul on spectra data using a CNN
    """

    def __init__(self, learning_rate: float = None):
        """
        Initializes and builds model

        :param learning_rate: Learning rate of the model, defaults to None
        :type learning_rate: int, optional
        """
        super().__init__(learning_rate=learning_rate)
        self.prediction_model = build_cnn()

    def train(
        self,
        training_data: Union[pd.DataFrame, np.ndarray],
        training_labels: Union[pd.Series, np.ndarray],
        epochs: int,
        validation_data: Optional[
            Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]
        ] = None,
    ) -> History:
        """
        Takes training data and labels and trains a model corresponding to the respective class.

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

        # pylint: disable=attribute-defined-outside-init
        # defined in super
        self.prediction_model, self.trainings_history = fit_cnn(
            self.prediction_model,
            training_data,
            training_labels,
            epochs=epochs,
            validation_data=validation_data,
            learning_rate=self.learning_rate,
        )

        return self.trainings_history

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> pd.Series:
        """
        Takes a dataframe or array and performs a prediction on it.
        The first axis is interpreted as number of samples

        :param data: DataFrame or array that contains the features for prediction
        :type data: Union[pd.DataFrame, np.ndarray]
        :return: Predictions, has the length data.shape[0]
        :rtype: pd.Series
        """
        return pd.Series(
            self.prediction_model.predict(reformat_spectra(data))[:, 0],
            name="rul_predictions",
        )

    @staticmethod
    def get_data_set_type() -> DataSetType:
        """
        Getter for the expected dataset type

        :return: Spectra Dataset type
        :rtype: DataSetType
        """
        return DataSetType.spectra


def load_class():
    """Returns the class in this file

    Returns:
        model_abc.AbstractModel
    """
    return CNNSpectraFeatures
