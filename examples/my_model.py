import numpy as np
import tensorflow as tf
from fl_models.abstract.abstract_model import AbstractModel


class MyModel(AbstractModel):
    def __init__(self, input_shape, learning_rate=None):
        super().__init__(learning_rate=learning_rate)

        self.prediction_model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(input_shape)),
                tf.keras.layers.Dense(4),
                tf.keras.layers.Dense(1),
            ]
        )

    def predict(self, data):
        features = data if isinstance(data, np.ndarray) else data.to_numpy()

        return self.prediction_model.predict(features)

    def train(self, training_data, training_labels, epochs, validation_data=None):
        val_tuple = validation_data
        if validation_data is not None:
            # Convert to numpy
            val_tuple = (
                val_tuple[0]
                if isinstance(val_tuple[0], np.ndarray)
                else val_tuple[0].to_numpy(),
                val_tuple[1]
                if isinstance(val_tuple[1], np.ndarray)
                else val_tuple[1].to_numpy(),
            )

        # Set optimizer and loss
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate if self.learning_rate is not None else 0.2
        )

        self.prediction_model.compile(loss="mse", optimizer=optimizer)

        # Train model
        return self.prediction_model.fit(
            training_data
            if isinstance(training_data, np.ndarray)
            else training_data.to_numpy(),
            training_labels
            if isinstance(training_labels, np.ndarray)
            else training_labels.to_numpy(),
            epochs=epochs,
            validation_data=val_tuple,
            verbose=1,
        )


def load_class():
    return MyModel
