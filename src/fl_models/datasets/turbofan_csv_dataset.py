"""
Contains interfaces to load the Turbofan dataset from a csv file
"""
from typing import List, Optional
import pandas as pd

from fl_models.abstract.abstract_turbofan_dataset import AbstractTurbofanDataset


# pylint: disable=too-few-public-methods
# Due to interface
class TurbofanCSVDataset(AbstractTurbofanDataset):
    """
    Class that handles loading of the Turbofan dataset from a csv file.

    For more see: A. Saxena, K. Goebel, D. Simon, and N. Eklund, Damage Propagation Modeling for \
        Aircraft Engine Run-to-Failure Simulation, in the Proceedings of the 1st International \
        Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.
    """

    def __init__(
        self,
        file_path: str,
        remaining_sensors: Optional[List[str]] = None,
        file_id: Optional[int] = None,
        min_max_norm_columns: Optional[List[str]] = None,
    ) -> None:
        """
        Loads the turbofan dataset from the specified csv file

        :param file_path: Path to csv file
        :type file_path: str
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
        super().__init__(
            remaining_sensors=remaining_sensors,
            file_id=file_id,
            min_max_norm_columns=min_max_norm_columns,
        )

        self._file_path = file_path

    def get_dataframe(self) -> pd.DataFrame:
        """
        Loads the specified csv file from self._file_path, filters out all sensors except the ones
        specified in self._remaining_sensors, the target column ("rul"), and the "unit_num" column.
        If specified, filters out the data rows with a file id other than self._file_id and
        normalizes the data. Columns in self._not_normal_dist are normalized using a
        min max scaler, the rest using a standard scaler.

        :return: Loaded dataframe
        :rtype: pd.DataFrame
        """
        print("Reading from csv file...")
        df: pd.DataFrame = pd.read_csv(self._file_path)

        # drop the first column with the row number
        if "Unnamed: 0" in df.columns:
            # pylint: disable=no-member
            # Pylint mistypes df and therefore thinks it does not have a drop function
            df.drop("Unnamed: 0", axis=1, inplace=True)

        return self._filter_dataframe(df)


def load_class():
    """Getter for Dynamic Loader

    :return: Class contained in this file
    """
    return TurbofanCSVDataset
