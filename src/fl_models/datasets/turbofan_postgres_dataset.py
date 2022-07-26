"""
Contains interfaces to load the Turbofan dataset from a postgres database
"""
from typing import Optional, List
import pandas as pd
import sqlalchemy as sqla

from fl_models.abstract.abstract_turbofan_dataset import AbstractTurbofanDataset


# pylint: disable=too-few-public-methods
# Due to interface
class TurbofanPostgresDataset(AbstractTurbofanDataset):
    """
    Class that handles loading of the Turbofan dataset from a database

    For more see: A. Saxena, K. Goebel, D. Simon, and N. Eklund, Damage Propagation Modeling for \
        Aircraft Engine Run-to-Failure Simulation, in the Proceedings of the 1st International \
        Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        user_name: str = None,
        table_name: str = None,
        remaining_sensors: Optional[List[str]] = None,
        file_id: Optional[int] = None,
        min_max_norm_columns: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """
        Loads the turbofan dataset from the specified table in the specified postgres database

        :param user_name: (Required) User name used to connect to database, defaults to None
        :type user_name: str
        :param table_name: (Required) Name of the table that should be used, defaults to None
        :type table_name: str
        :param password: Password used to connect to the database, defaults to None
        :type password: str, optional
        :param host: Host IP of the database, defaults to "127.0.0.1"
        :type host: str, optional
        :param port: Port on which the database is located, defaults to "5432"
        :type port: str, optional
        :param remaining_sensors: Column filter, defaults to None. Only the specified columns, \
            "rul", and "unit_num" remain. If None no filter is applied.
        :type remaining_sensors: Optional[List[str]], optional
        :param file_id: Row filter, defaults to None. Only data rows with the matching file_id \
            remain.
        :type file_id: Optional[int], optional
        :param min_max_norm_columns: List of column that will be scaled using a min max scaler \
            instead of a standard one, defaults to None
        :type min_max_norm_columns: Optional[List[str]], optional
        """
        super().__init__(
            remaining_sensors=remaining_sensors,
            file_id=file_id,
            min_max_norm_columns=min_max_norm_columns,
        )

        assert (
            user_name is not None
        ), "A user name is required to connect to the database!"
        assert (
            table_name is not None
        ), "The name of the table is required to connect to the database!"

        self._table_name = table_name

        host = kwargs.get("host", "127.0.0.1")
        port = kwargs.get("port", "5432")

        password = kwargs.get("password")
        if password is None:
            password = ""
        else:
            password = ":" + password

        self.database = f"postgresql://{user_name}{password}@{host}:{port}/turbofan"

        self._engine = sqla.create_engine(self.database, echo=False)

    def get_dataframe(self) -> pd.DataFrame:
        """
        Loads the dataset from the specified database, filters out all sensors except the ones
        specified in self._remaining_sensors, the target column ("rul"), and the "unit_num" column,
        if specified, filters out the data rows with a file id other than self._file_id and
        normalizes the data. Columns in self._not_normal_dist are normalized using a
        min max scaler, the rest using a standard scaler.

        :return: Loaded dataframe
        :rtype: pd.DataFrame
        """

        print("Reading from database...")
        metadata = sqla.MetaData(bind=None)

        table = sqla.Table(
            self._table_name, metadata, autoload=True, autoload_with=self._engine
        )

        query = sqla.select([table])

        df = pd.read_sql(query, con=self._engine)

        return self._filter_dataframe(df)


def load_class():
    """Getter for Dynamic Loader

    :return: Class contained in this file
    """
    return TurbofanPostgresDataset
