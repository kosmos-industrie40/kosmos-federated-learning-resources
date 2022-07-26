"""
Module containing different loss functions for the LSTM
"""
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss


class RootMeanSquaredError(Loss):
    """
    Custom loss function for RMSE
    """

    def call(
        self,
        y_true,
        y_pred,
    ):
        """
        Calculated RMSE

        :param y_true: the label
        :param y_pred: the prediction

        :return: RMSE
        """
        return tf.sqrt(tf.reduce_mean((y_true - y_pred) ** 2))


class WeightedRootMeanSquaredError(Loss):
    """
    Custom loss function for a weighted RMSE
    """

    def call(self, y_true, y_pred):
        """
        Calculated RMSE multiplied by weights alpha and beta

        :param y_true: the label
        :param y_pred: the prediction

        :return: Weighted RMSE
        """
        alpha = 2
        beta = 1
        rmse = tf.sqrt(tf.reduce_mean((y_true - y_pred) ** 2))
        if rmse > 0:
            # rul >= 0 is good
            rmse *= alpha
        else:
            # rul < 0 is bad
            rmse *= beta
        return rmse


class ScoringFunction(Loss):
    """
    Weighted loss function according to paper X
    """

    @staticmethod
    def calculate_weight(diff):
        """
        Defines weight depending on the difference
        between target and label

        :param diff: Difference between target and label
        :return: Exponential of input / weight
        """
        i = 13 if diff < 0 else 10
        return tf.math.minimum(
            tf.exp(tf.truediv(diff, tf.cast(i, tf.float32))) - 1, tf.constant(np.inf)
        )

    def call(self, y_true, y_pred):
        """
        Maps the defined weighting to the input
        Sum_i=1^n e^(Y_i/13)-1, Y_i < 0
        Sum_i=1^n e^(Y_i/10)-1, Y_i >= 0

        :param y_true: the label
        :param y_pred: the prediction

        :return: Weighted loss
        """
        return tf.math.minimum(
            tf.reduce_sum(
                tf.map_fn(ScoringFunction.calculate_weight, (y_true - y_pred))
            ),
            tf.constant(np.inf),
        )


def map_loss_name(loss_name: str) -> Loss:
    """
    Maps a string to the corresponding class

    :param loss_name: name of the loss function

    :return: Loss Function

    """

    # pylint: disable=no-else-return, too-many-return-statements
    if loss_name.lower() == "huber":
        return tf.keras.losses.Huber()
    elif loss_name.lower() == "mse":
        return tf.keras.losses.MeanSquaredError()
    elif loss_name.lower() == "msle":
        return tf.keras.losses.MeanSquaredLogarithmicError()
    elif loss_name.lower() == "rootmeansquarederror" or loss_name.lower() == "rmse":
        return RootMeanSquaredError()
    elif (
        loss_name.lower() == "weightedrootmeansquarederror"
        or loss_name.lower() == "wrmse"
    ):
        return WeightedRootMeanSquaredError()
    elif loss_name == "scoringfunction":
        return ScoringFunction()
    else:
        logging.warning("Not an existing loss method")
        logging.warning("Fallback to default RMSE")
        return RootMeanSquaredError()
