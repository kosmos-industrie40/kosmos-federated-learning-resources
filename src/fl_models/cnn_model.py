"""
Util class for cnn model creation and fitting
"""
from typing import Tuple, Union
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.callbacks import History
from fl_models.util.constants import SPECTRA_SHAPE


def build_cnn(input_shape: Tuple[int] = SPECTRA_SHAPE) -> Model:
    """
    Creates a CNN for RUL using spectra

    :param input_shape: Input shape of the spectra, defaults to SPECTRA_SHAPE
    :type input_shape: tuple, optional
    :return: Tensorflow model
    :rtype: Model
    """
    input_img = Input(shape=input_shape)
    conv_1 = layers.Conv2D(10, (6, 6))(input_img)
    max_pool_1 = layers.MaxPool2D((2, 2))(conv_1)
    max_pool_1 = layers.BatchNormalization()(max_pool_1)
    conv_2 = layers.Conv2D(10, (6, 6))(max_pool_1)
    max_pool_2 = layers.MaxPool2D((2, 2))(conv_2)
    max_pool_2 = layers.BatchNormalization()(max_pool_2)
    cnn = layers.Flatten()(max_pool_2)

    ffnn = layers.Dense(512, activation="relu")(cnn)
    ffnn = layers.Dropout(0.5)(ffnn)
    ffnn = layers.Dense(512, activation="relu")(ffnn)
    ffnn = layers.Dropout(0.5)(ffnn)
    ffnn = layers.Dense(1)(ffnn)

    return Model(input_img, ffnn)


def reformat_spectra(data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Reshapes the incoming 1D data to 3D

    :param data: Data that has the shape (batch_size, spectra_shape[0] * spectra_shape[1] * \
        spectra_shape[2])
    :type data: Union[pd.DataFrame, np.ndarray]
    :return: Reshaped array of shape (batch_size, spectra_shape[0], spectra_shape[1], \
        spectra_shape[2])
    :rtype: np.ndarray
    """
    numpy_arr = data if isinstance(data, np.ndarray) else data.to_numpy()
    return numpy_arr.reshape((data.shape[0], *SPECTRA_SHAPE))


# pylint: disable=too-many-arguments
def fit_cnn(
    cnn_model: Model,
    training_data: Union[pd.DataFrame, np.ndarray],
    training_labels: pd.Series,
    epochs: int = 2,
    validation_data=(None, None),
    learning_rate: float = None,
) -> Tuple[Model, History]:
    """
    Creates a CNN model and fits it with the given training data

    :param cnn_model: Model that will be trained
    :type cnn_model: Model
    :param training_data: Training data (Spectra)
    :type training_data: Union[pd.DataFrame, np.ndarray]
    :param training_labels: Training labels
    :type training_labels: pd.Series
    :param epochs: Number of epochs to train, defaults to 2
    :type epochs: int, optional
    :param validation_data: Tuple (validation data, validation labels), defaults to (None, None)
    :type validation_data: tuple, optional
    :param learning_rate: Learning rate, defaults to None
    :type learning_rate: float, optional
    :return: The trained model and training history
    :rtype: Tuple[Model, History]
    """

    if learning_rate is None:
        optimizer = keras.optimizers.Adam()
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.MeanSquaredError()
    cnn_model.compile(optimizer=optimizer, loss=loss)
    if validation_data is None or validation_data == (None, None):
        training_history = cnn_model.fit(
            x=reformat_spectra(training_data),
            y=training_labels,
            epochs=epochs,
            verbose=2,
        )
    else:
        training_history = cnn_model.fit(
            x=reformat_spectra(training_data),
            y=training_labels,
            epochs=epochs,
            verbose=2,
            validation_data=reformat_spectra(validation_data),
        )
    return cnn_model, training_history
