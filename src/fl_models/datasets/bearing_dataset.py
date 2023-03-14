"""
Load the bearing dataset from inovex nextcloud
"""
import os
import zipfile
import logging
from typing import List, Union

import pandas as pd
import wget
from tqdm import tqdm


from fl_models.abstract.abstract_dataset import AbstractDataset


# pylint: disable=too-few-public-methods
# Due to interface
class BearingDataset(AbstractDataset):
    """
    Class that handles downloading and loading of the bearing dataset
    """

    def __init__(self, file_name: str = None, filter_list: Union[None, List] = None):
        """
        Loads dataset from inovex nextcloud

        :param file_name: Name of the target files, either spectra.csv or features.csv, defaults \
            to None
        :type file_name: str, optional
        :param filter_list: List of bearing that should be loaded, defaults to None -> all bearings
        :type filter_list: Union[None, List], optional
        """
        super().__init__()
        assert (
            file_name is not None
        ), "Missing parameter 'file_name'. Cannot decide whether feature or spectra data should \
            be loaded!"

        self.file_name = file_name
        self.filter_list = filter_list
        self._path = BearingDataset._get_bearing_processed_data_path()

    @staticmethod
    def _get_bearing_processed_data_path() -> str:
        """
        Download dataset and unzip

        :return: Local path to downloaded data
        :rtype: str
        """
        print("Preparing data ...")
        url = "https://artifactory.inovex.de/artifactory/KOSMos-public/processed_bearing_\
            data.zip"
        tmp_folder_path = "/tmp"
        data_file_name = "processed_bearing_data"
        data_file_extension = "zip"

        # Check if temp folder exists, if not create
        if not os.path.exists(tmp_folder_path) or not os.path.isdir(tmp_folder_path):
            os.mkdir(tmp_folder_path)

        path = os.path.join(tmp_folder_path, f"{data_file_name}")
        if not os.path.exists(path):
            print("Downloading bearing processed data. This could take a while ...")
            wget.download(url, tmp_folder_path)
            with zipfile.ZipFile(f"{path}.{data_file_extension}", "r") as zip_ref:
                zip_ref.extractall(tmp_folder_path)
            os.remove(f"{path}.{data_file_extension}")

        logging.info("Bearing processed data found.")
        return path

    def get_dataframe(self) -> pd.DataFrame:
        """
        Reads all CSVs and compiles them into a data frame of a given sub set and CSV type
        :return: list of read and compiled data frames
        """

        bearing_list = []

        # Get all bearings
        for _, directories, _ in os.walk(self._path):
            bearing_list = directories
            break  # don't walk subdirs

        bearing_list = sorted(list(bearing_list))
        if self.filter_list is not None:
            bearing_list = list(filter(lambda x: x in self.filter_list, bearing_list))

        full_dataframe: pd.DataFrame = None

        for bearing in tqdm(
            bearing_list,
            desc=f"Reading computed features of bearings from: {self.file_name}",
        ):

            current_df = pd.read_csv(os.path.join(self._path, bearing, self.file_name))
            current_df["bearing_name"] = bearing

            if full_dataframe is None:
                full_dataframe = current_df
            else:
                full_dataframe = pd.concat([full_dataframe, current_df])

        return full_dataframe


def load_class():
    """Returns the class in this file

    Returns:
        abstract.abstract_dataset.AbstractDataset
    """
    return BearingDataset
