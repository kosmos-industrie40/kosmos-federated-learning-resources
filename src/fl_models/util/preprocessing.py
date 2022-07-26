"""
Util module for data preprocessing
"""
from typing import List
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def standard_normalization(data: pd.DataFrame, column_list: List[str]):
    """
    Normalizes given columns in the dataframe using a standard scaler

    :param data: DataFrame containing the input data
    :param column_list: List containing the column names that will be normalized

    :return: A deep copy of data where the columns in column_list have been normalized
        using a standard scaler
    """
    scaler = StandardScaler()
    scaler.fit(data.loc[:, column_list])

    norm_data = data.copy(deep=True)
    norm_data.loc[:, column_list] = scaler.transform(data.loc[:, column_list])

    return norm_data


def min_max_normalization(data: pd.DataFrame, column_list: List[str]):
    """
    Normalizes given columns in the dataframe using a min-max scaler

    :param data: DataFrame containing the input data
    :param column_list: List containing the column names that will be normalized

    :return: A deep copy of data where the columns in column_list have been normalized
        using a min-max scaler
    """
    scaler = MinMaxScaler()
    scaler.fit(data.loc[:, column_list])

    norm_data = data.copy(deep=True)
    norm_data.loc[:, column_list] = scaler.transform(data.loc[:, column_list])

    return norm_data
