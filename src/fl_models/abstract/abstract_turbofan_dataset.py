"""
Contains common interfaces to loading the turbofan dataset
"""
from typing import List, Optional
from abc import ABC, abstractmethod

import pandas as pd

from fl_models.abstract.abstract_dataset import AbstractDataset
from fl_models.util.preprocessing import standard_normalization, min_max_normalization


# pylint: disable=too-few-public-methods
# Due to interface
class AbstractTurbofanDataset(AbstractDataset, ABC):
    """
    Class that handles loading of the Turbofan dataset from a csv file
    """

    @abstractmethod
    def __init__(
        self,
        remaining_sensors: Optional[List[str]] = None,
        file_id: Optional[int] = None,
        min_max_norm_columns: Optional[List[str]] = None,
    ) -> None:
        """
        Abstract interface for loading the turbofan dataset

        :param remaining_sensors: Column filter, defaults to None. Only the specified columns and \
            "rul" and "unit_num" remain. If None no filter is applied.
        :type remaining_sensors: Optional[List[str]], optional
        :param file_id: Row filter, defaults to None. Only data rows with the matching file_id \
            remain.
        :type file_id: Optional[int], optional
        :param min_max_norm_columns: List of columns that will be scaled using a min max scaler \
            instead of a standard one, defaults to None
        :type min_max_norm_columns: Optional[List[str]], optional
        """
        super().__init__()

        self._remaining_sensors = remaining_sensors

        self._meta_cols = ["unit_num", "rul"]

        self.extended_remaining_sensors = None

        if self._remaining_sensors is not None:
            self.extended_remaining_sensors = []
            self.extended_remaining_sensors.extend(self._remaining_sensors)
            self.extended_remaining_sensors.extend(self._meta_cols)

        self._min_max_norm_columns = (
            min_max_norm_columns if min_max_norm_columns else []
        )

        self._file_id = file_id

    def _filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters and preprocesses the given dataframe based on the self._file_id, \
        self._remaining_sensors and self._min_max_norm_columns properties

        :param df: Dataframe that will be processed
        :type df: pd.DataFrame

        :return: Loaded dataframe
        :rtype: pd.DataFrame
        """

        if self._file_id:
            df = df[df["file_id"] == self._file_id]

        if self._remaining_sensors is None:
            self._remaining_sensors = list(df)
            self.extended_remaining_sensors = self._remaining_sensors

        df = df[self.extended_remaining_sensors]

        # Use standard normalization on remaining sensors
        df = standard_normalization(
            df,
            list(
                filter(
                    lambda x: x not in self._min_max_norm_columns,
                    self._remaining_sensors.copy(),
                )
            ),
        )

        # Use min max scaling on selected sensors
        if len(self._min_max_norm_columns) > 0:
            df = min_max_normalization(df, self._min_max_norm_columns)
        # Scale
        return df
