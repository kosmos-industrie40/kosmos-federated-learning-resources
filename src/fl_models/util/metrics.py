"""
Contains functions for metric calculation
"""
from typing import Callable
from math import sqrt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from numpy import std

from fl_models.util.constants import (
    MSE_KEY,
    R2_SCORE_KEY,
    RMSE_KEY,
    CORR_COEFF_KEY,
    STANDARD_DEVIATION_KEY,
)


def rename(newname: str) -> Callable[[Callable], Callable]:
    """
    Decorator that renames a function

    :param newname: New name of the function
    :type newname: str
    """

    def decorator(func):
        func.__name__ = newname
        return func

    return decorator


def accuracy(y_actual, y_predicted):
    """
    Calculates the ratio of correctly predicted

    :param y_actual: Correct values
    :param y_predicted: Predicted values
    :return: root mean square
    """
    return sqrt(mean_squared_error(y_actual, y_predicted))


@rename(MSE_KEY)
def mse(y_actual, y_predicted):
    """
    Calculates the mean squared error

    :param y_actual: Correct values
    :param y_predicted: Predicted values
    :return: mean squared error
    """
    return mean_squared_error(y_actual, y_predicted)


@rename(R2_SCORE_KEY)
def score(y_actual, y_predicted):
    """
    Calculates the r2 score

    :param y_actual: Correct values
    :param y_predicted: Predicted values
    :return: r2 score
    """
    return r2_score(y_actual, y_predicted)


@rename(RMSE_KEY)
def rmse(y_actual, y_predicted):
    """
    Calculates sqrt(1/n * sum((y_actual-y_predicted)^2))

    :param y_actual: Correct values
    :param y_predicted: Predicted values
    :return: root mean square
    """
    return sqrt(mean_squared_error(y_actual, y_predicted))


def mae(y_actual, y_predicted):
    """
    Calculates 1/n * sum(|y_actual-y_predicted|)

    :param y_actual: Correct values
    :param y_predicted: Predicted values
    :return: root mean square
    """
    return mean_absolute_error(y_actual, y_predicted)


@rename(CORR_COEFF_KEY)
def correlation_coefficient(y_actual, y_predicted):
    """
    Calculates the pearson correlation coefficient

    :param y_actual: Correct values
    :param y_predicted: Predicted values
    :return: Correlation coefficient
    """
    coeff, _ = pearsonr(y_actual, y_predicted)
    return coeff


@rename(STANDARD_DEVIATION_KEY)
def standard_deviation(value_list):
    """
    Calculates the standard deviation in the given list

    :param value_list: Values
    :return: Standard deviation
    """
    return std(value_list)
