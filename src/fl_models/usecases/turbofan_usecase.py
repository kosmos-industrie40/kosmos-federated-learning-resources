"""
Module for handling Bearing Dataset. Loads Bearing Dataset and creates evaluation
"""

from typing import Any, Iterable, Optional, List, Union
import mlflow
import numpy as np
import pandas as pd

from fl_models.util.metrics import rmse, mse, score
from fl_models.util.dynamic_loader import load_model, load_dataset

from fl_models.abstract.abstract_dataset import AbstractDataset
from fl_models.abstract.abstract_model import AbstractModel

from fl_models.abstract.abstract_usecase import FederatedLearningUsecase


class TurbofanUseCase(FederatedLearningUsecase):
    """
    Implements the Turbofan federated learning usecase
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        file_path: str = None,
        user_name: str = None,
        sequence_length: int = 30,
        log_mlflow: bool = True,
        remaining_sensors: Optional[List[str]] = None,
        file_id: Optional[int] = None,
        min_max_norm_columns: Optional[List[str]] = None,
        loss: str = "WeightedRootMeanSquaredError",
        **kwargs,
    ) -> None:
        """
        Usecase for processing the NASA Turbofan dataset. The dataset can either be loaded from
        a csv file or a postgres database

        :param file_path: Path of the CSV file containing the dataset, defaults to None
        :type file_path: str, optional
        :param user_name: User name used to access the database, defaults to None
        :type user_name: str, optional
        :param sequence_length: Sequence length used for prediction/training, defaults to 30
        :type sequence_length: int, optional
        :param log_mlflow: If true, model parameters and metrics are logged to mlflow,
            defaults to True
        :type log_mlflow: bool, optional
        :param remaining_sensors: List of sensors that will be used for inference, defaults to None
        :type remaining_sensors: Optional[List[str]], optional
        :param file_id: File id used to filter dataset, defaults to None
        :type file_id: Optional[int], optional
        :param min_max_norm_columns: Columns that will use min max scaling instead of
            standard deviation, defaults to None
        :type min_max_norm_columns: Optional[List[str]], optional
        :param loss: Name of the loss function, defaults to "WeightedRootMeanSquaredError"
        :type loss: str, optional
        :raises ValueError: If neither file_name nor user_name are defined
        """

        super().__init__(log_mlflow)

        # Init dataset
        if "csv" in self.get_dataset_name().lower():

            if file_path is None:
                raise ValueError('Use Case is missing "file_path" parameter!')

            self.dataset: AbstractDataset = load_dataset(
                self.get_dataset_name(),
                file_path=file_path,
                remaining_sensors=remaining_sensors,
                min_max_norm_columns=min_max_norm_columns,
                file_id=file_id,
            )
        else:

            if user_name is None:
                raise ValueError('Use Case is missing "user_name" parameter!')

            self.dataset: AbstractDataset = load_dataset(
                self.get_dataset_name(),
                user_name=user_name,
                remaining_sensors=remaining_sensors,
                min_max_norm_columns=min_max_norm_columns,
                file_id=file_id,
                table_name=kwargs.pop("table_name", None),
                password=kwargs.pop("password", None),
                host=kwargs.pop("host", None),
                port=kwargs.pop("port", None),
            )

        self.number_of_sensors = -1  # Set in load_data
        self.sequence_length = sequence_length
        self.metrics = [rmse, mse, score]

        self.num_data_rows = 0
        self.load_data()

        self.abstract_model: AbstractModel = load_model(
            self.get_model_name(),
            input_shape=(sequence_length, self.number_of_sensors),
            loss=loss,
            learning_rate=kwargs.pop("learning_rate", None),
        )

        # Log model
        if log_mlflow:
            mlflow.log_param("input_shape", (sequence_length, self.number_of_sensors))
            mlflow.log_param("sequence_length", sequence_length)
            mlflow.log_param("file_id", file_id)

        # Warn in case there are unused arguments
        self.check_for_unused_args(kwargs)

    def get_data(self, flat: bool = False) -> Iterable[Union[Any, np.ndarray]]:
        """
        Getter for usecase specific data

        :param flat: If False, the data will be grouped by identifier, if True the data is not
            grouped, defaults to False
        :type flat: bool, optional
        :return: Feature data for training/evaluation
        :rtype: Iterable[Union[Any, np.ndarray]]
        """
        if not flat:
            return self.data
        return np.concatenate(self.data)

    def get_labels(self, flat: bool = False) -> Iterable[Union[Any, np.ndarray]]:
        """
        Getter for the usecase specific labels

        :param flat: If False, the labels will be grouped by identifier, defaults to False
        :type flat: bool, optional
        :return: Labels for training/evaluation
        :rtype: Iterable[Union[Any, np.ndarray]]
        """
        if not flat:
            return self.labels
        return np.concatenate(self.labels)

    def _reshape_dataframe_to_3d_array_list(
        self, dataframe: pd.DataFrame
    ) -> List[np.ndarray]:
        """
        Reshapes the dataframe to sequences of sequence length. The dataframe is grouped by
        the "unit_number".
        Each of these groups is transformed into sequences. Sequences are filled if necessary.

        :param dataframe: Dataframe that will be transformed
        :type dataframe: pd.DataFrame
        :return: A list of 3D arrays
        :rtype: List[np.ndarray]
        """

        unit_list = dataframe["unit_num"].unique()
        dataframe = dataframe.groupby("unit_num", as_index=False)

        labels_list = []
        data_list = []
        self.num_data_rows = 0

        for current_unit in unit_list:
            current_df = dataframe.get_group(current_unit).drop("unit_num", axis=1)
            labels = current_df["rul"].to_numpy()
            current_df = current_df.drop("rul", axis=1)

            unit_arr = current_df.to_numpy()

            num_patches = int(np.ceil(unit_arr.shape[0] / self.sequence_length))

            # fill array to match sequence length
            padded_arr = np.full(
                shape=(self.sequence_length * num_patches, unit_arr.shape[1]),
                fill_value=-99.0,
            )

            padded_arr[0 : unit_arr.shape[0], :] = unit_arr

            padded_labels_arr = np.empty(self.sequence_length * num_patches)

            padded_labels_arr[0 : len(labels)] = labels

            if padded_labels_arr.shape[0] > labels.shape[0]:
                padded_labels_arr[len(labels) :] = (
                    np.arange(0, padded_labels_arr.shape[0] - labels.shape[0])
                    + labels[-1]
                )[::1]

            reshaped_arr = np.reshape(
                padded_arr, (num_patches, self.sequence_length, unit_arr.shape[1])
            )
            reshaped_labels = padded_labels_arr[
                self.sequence_length - 1 :: self.sequence_length
            ]

            self.num_data_rows += reshaped_arr.shape[0]
            data_list.append(reshaped_arr)
            labels_list.append(reshaped_labels)

        return unit_list.tolist(), data_list, labels_list

    def load_data(self):
        """
        Loads the bearing data used for training in memory further to train the model
        """

        full_df = self.dataset.get_dataframe()
        self.number_of_sensors = full_df.shape[1] - 2

        # Do sequence transform
        (
            self.identifiers,
            self.data,
            self.labels,
        ) = self._reshape_dataframe_to_3d_array_list(full_df)

    @staticmethod
    def get_model_name() -> str:
        """
        Getter for the model name associated with this usecase

        :return: Modelname
        :rtype: str
        """
        return "LSTMModel"

    @staticmethod
    def get_dataset_name() -> str:
        """
        Getter for the dataset name associated with this usecase

        :return: Datasetname
        :rtype: str
        """
        return "TurbofanCSVDataset"

    def get_number_of_samples(self) -> int:
        """
        Getter for the number of samples in the dataset

        :return: Number of samples
        :rtype: int
        """
        return self.num_data_rows


def load_class() -> TurbofanUseCase:
    """
    Getter for Dynamic Loader
    """
    return TurbofanUseCase
