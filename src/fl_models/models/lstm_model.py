"""
Contains a LSTM for RUL prediction
"""
from typing import Optional, Tuple

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking

from fl_models.abstract.abstract_model import AbstractModel
from fl_models.util.losses import map_loss_name


class LSTMModel(AbstractModel):
    """
    LSTM Model for RUL prediction

    :param parameters: Contains model parameters like the input_shape
    :type parameters: Dict[str, Any]
    :param batch_size: The batch size used during training
    :type batch_size: int
    :param prediction_model: The LSTM Model used for prediction
    :type prediction_model: Sequential
    """

    def __init__(
        self,
        input_shape: Tuple[int],
        loss: str = "RMSE",
        learning_rate: float = None,
        **kwargs,
    ):
        """
        Initializes the model

        :param input_shape: Shape of the input. Usually (sequence length, number of features)
        :param loss: Name of the utilized loss function. Possible values are: `"Huber"`, `"MSE"`, \
            `"MSLE"`, `"RMSE"`, `"WeightedRootMeanSquaredError"` and `"ScoringFunction"`
        :param kwargs: Further model configurations
        :param learning_rate: Learning rate of the model, defaults to None
        :type learning_rate: float, optional
        """
        super().__init__(learning_rate=learning_rate)

        self.parameters = {
            "input_shape": input_shape,
            "nodes_per_layer": kwargs.get("nodes_per_layer", [256]),
            "dropout": kwargs.get("dropout", 0.1),
            "activation": kwargs.get("activation", "sigmoid"),
            "loss": loss,
        }

        self.batch_size = kwargs.get("batch_size", 128)

        self.prediction_model = self._create_model()

    def _create_model(self) -> tf.keras.Sequential:
        """
        Creates the model with the defined parameters
        Returns
        -------
        tensorflow.keras.model
        """
        model = Sequential()
        model.add(Masking(mask_value=-99.0, input_shape=self.parameters["input_shape"]))
        if len(self.parameters["nodes_per_layer"]) <= 1:
            model.add(
                LSTM(
                    self.parameters["nodes_per_layer"][0],
                    activation=self.parameters["activation"],
                )
            )
            model.add(Dropout(self.parameters["dropout"]))
        else:
            model.add(
                LSTM(
                    self.parameters["nodes_per_layer"][0],
                    activation=self.parameters["activation"],
                    return_sequences=True,
                )
            )
            model.add(Dropout(self.parameters["dropout"]))
            model.add(
                LSTM(
                    self.parameters["nodes_per_layer"][1],
                    activation=self.parameters["activation"],
                )
            )
            model.add(Dropout(self.parameters["dropout"]))
        model.add(Dense(1))

        return model

    def predict(self, data: np.ndarray) -> pd.Series:
        """
        Performs a prediction on the given data

        :param data: Features for the prediction
        :type data: np.ndarray
        :return: Predicted RUL
        :rtype: np.ndarray
        """
        assert isinstance(data, np.ndarray), (
            f"Unsupported input type {data.__class__.__name__}! Currently, only numpy arrays are "
            + "supported."
        )
        return self.prediction_model.predict(data)

    def train(
        self,
        training_data: np.ndarray,
        training_labels: np.ndarray,
        epochs: int = None,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> History:
        """
        Takes training data and labels and trains a model corresponding to the respective class.
        Also sets the prediction model.

        :param training_data: DataFrame or array of features. First axis is interpreted as number of
            samples
        :type training_data: np.ndarray
        :param training_labels: Series of labels. Has to be the same length as training_data
        :type training_labels: np.ndarray
        :param validation_data: Tuple containing (validation features, validation labels). \
            DataFrame or array of features used for validation. First axis is \
            interpreted as number of samples
        :type validation_data: Tuple[np.ndarray, np.ndarray], optional
        :param epochs: Number of training epochs
        :type epochs: int
        :return: Returns the trainings history. Is expected to contain a list of losses under
            history.history["loss"]
        :rtype: History
        """
        assert isinstance(training_data, np.ndarray), (
            f"Unsupported input type {training_data.__class__.__name__} "
            + "for 'training_data'! Currently, only numpy arrays are supported."
        )
        assert isinstance(training_labels, np.ndarray), (
            f"Unsupported input type {training_labels.__class__.__name__} "
            + "for 'training_labels'! Currently, only numpy arrays are supported."
        )

        if validation_data is not None:
            assert isinstance(validation_data[0], np.ndarray) and isinstance(
                validation_data[1], np.ndarray
            ), (
                "Unsupported input type for 'validation_data'! Currently, only numpy arrays are"
                + " supported."
            )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate if self.learning_rate is not None else 0.2
        )

        loss = map_loss_name(self.parameters["loss"])
        self.prediction_model.compile(loss=loss, optimizer=optimizer)

        return self.prediction_model.fit(
            training_data,
            training_labels,
            epochs=epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            verbose=1,
        )


def load_class():
    """
    Getter for Dynamic Loader
    """
    return LSTMModel
