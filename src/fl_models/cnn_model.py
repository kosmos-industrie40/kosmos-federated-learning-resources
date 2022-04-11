import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, Input, Model
from fl_models.util.constants import SPECTRA_SHAPE


def build_cnn(input_shape: tuple = SPECTRA_SHAPE) -> Model:
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


def fit_cnn(
    x: np.array,
    y: pd.Series,
    epochs: int = 2,
    input_shape=(129, 21, 2),
    validation_data=(None, None),
    learning_rate: float = None,
):

    cnn_model = build_cnn(input_shape)
    if learning_rate is None:
        optimizer = keras.optimizers.Adam()
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.MeanSquaredError()
    cnn_model.compile(optimizer=optimizer, loss=loss)
    if validation_data == (None, None):
        training_history = cnn_model.fit(x=x, y=y, epochs=epochs, verbose=2)
    else:
        training_history = cnn_model.fit(
            x=x, y=y, epochs=epochs, verbose=2, validation_data=validation_data,
        )
    return cnn_model, training_history


if __name__ == "__main__":
    model = build_cnn((129, 21, 2))
    print(model.summary())
