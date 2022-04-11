import os
import pandas as pd
from typing import Dict, List
from tqdm import tqdm
import wget
import zipfile
import logging

from fl_models.DataSetType import DataSetType
from fl_models.util.helper_methods import concat_dfs
from fl_models.util.constants import SPECTRA_CSV_NAME


def get_bearing_processed_data_path():
    print("Preparing data ...")
    url = "https://nextcloud.inovex.de/index.php/s/rAoGRdnqHHHe82i/download/processed_bearing_data.zip"
    tmp_folder_path = '/tmp'
    data_file_name = 'processed_bearing_data'
    data_file_extension = 'zip'

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


def read_feature_dfs_as_dict(
    bearing_names: List[str], data_set_type: DataSetType = DataSetType.spectra,
) -> Dict[str, pd.DataFrame]:
    """
    Reads all CSVs and compiles them into a data frame of a given sub set and CSV type
    :param bearing_names: List consisting of BearingX_X denoting which bearings should be loaded
    :param data_set_type: type of features that are to be read
    :return: list of read and compiled data frames
    """
    path = get_bearing_processed_data_path()

    bearing_list = [
        name for name in bearing_names if os.path.isdir(os.path.join(path, name))
    ]
    bearing_list = sorted(bearing_list)
    file_name: str = data_set_type if isinstance(data_set_type, str) else data_set_type.value
    df_dict = {}
    for bearing in tqdm(
        bearing_list, desc="Reading computed features of bearings from: %s" % file_name
    ):
        df_dict[bearing] = pd.read_csv(os.path.join(path, bearing, file_name))
    return df_dict


def read_feature_dfs_as_dataframe(
    bearing_names: List[str], csv_name=SPECTRA_CSV_NAME
) -> (pd.DataFrame, pd.Series):
    """
    Reads all CSVs and compiles them into a data frame of a given sub set and CSV type
    :param bearing_names: names of bearings that are to be read from the data set subset
    :param csv_name: type of features that are to be read
    :return: list of read and compiled data frames
    """
    df_dict = read_feature_dfs_as_dict(bearing_names, data_set_type=csv_name)
    return df_dict_to_df_dataframe(df_dict)


def df_dict_to_df_dataframe(
    df_dict: Dict[str, pd.DataFrame]
) -> (pd.DataFrame, pd.Series):
    data = concat_dfs(df_dict)
    labels = data.pop("RUL")
    return data, labels
