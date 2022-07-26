"""
Module handling creation and training of feed forward neural networks (ffnn) for bearing rul
prediction using computed features
"""
from typing import Tuple
import pandas as pd
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.callbacks import History
from tensorflow.keras.optimizers import Adam


def build_ffnn_model(
    input_shape: tuple, dropout: bool, hidden_layers: int, hidden_units: int
) -> Model:
    """
    Creates a feed forward network

    :param input_shape: Input shape of the network
    :type input_shape: tuple
    :param dropout: Whether or not dropouts should be used
    :type dropout: bool
    :param hidden_layers: Number of hidden layers
    :type hidden_layers: int
    :param hidden_units: Number of units per hidden layer
    :type hidden_units: int
    :return: The resulting tensorflow model
    :rtype: Model
    """
    input_df = Input(shape=input_shape)

    ffnn = layers.Dense(hidden_units, activation="relu")(input_df)
    if dropout:
        ffnn = layers.Dropout(0.5)(ffnn)
    for _ in range(hidden_layers - 1):
        ffnn = layers.Dense(hidden_units, activation="relu")(ffnn)
        if dropout:
            ffnn = layers.Dropout(0.5)(ffnn)
    ffnn = layers.Dense(1)(ffnn)

    return Model(input_df, ffnn)


# pylint: disable=too-many-arguments
def fit_ffnn(
    ffnn: Model,
    training_data: pd.DataFrame,
    training_labels: pd.Series,
    epochs: int = 60,
    validation_data=(None, None),
    learning_rate: float = None,
) -> Tuple[Model, History]:
    """
    Takes the given ffnn and fits it to the given dataset

    :param ffnn: FFNN that will be fitted
    :type ffnn: Model
    :param training_data: DataFrame with features
    :type training_data: pd.DataFrame
    :param training_labels: Series with matching labels
    :type training_labels: pd.Series
    :param epochs: Number of training epochs, defaults to 60
    :type epochs: int, optional
    :param validation_data: Tuple (validation data, validation labels), defaults to (None, None)
    :type validation_data: tuple, optional
    :param learning_rate: Learing rate of the model, defaults to None
    :type learning_rate: float, optional
    :return: Tuple containing the new model and the training history
    :rtype: Tuple[Model, History]
    """
    if learning_rate is None:
        ffnn.compile(optimizer="adam", loss="mean_squared_error")
    else:
        ffnn.compile(
            optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error"
        )

    if validation_data is None or validation_data == (None, None):
        training_history = ffnn.fit(
            x=training_data, y=training_labels, epochs=epochs, shuffle=True, verbose=2
        )
    else:
        training_history = ffnn.fit(
            x=training_data,
            y=training_labels,
            epochs=epochs,
            shuffle=True,
            verbose=2,
            validation_data=validation_data,
        )
    return ffnn, training_history
