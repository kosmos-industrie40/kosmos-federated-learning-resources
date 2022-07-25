"""
Defines interface for datasets
"""
from abc import ABC, abstractmethod
import pandas as pd


# pylint: disable=too-few-public-methods
# Abstract interface, not meant to be instantiated
class AbstractDataset(ABC):
    """
    Abstract interface for dataset handling
    """

    @abstractmethod
    def get_dataframe(self) -> pd.DataFrame:
        """
        Loads all dataframes
        :returns: the loaded DataFrame
        """
        raise NotImplementedError()
