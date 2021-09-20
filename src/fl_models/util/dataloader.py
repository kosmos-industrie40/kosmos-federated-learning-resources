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
    #url = "http://s3-de-central.profitbricks.com/bearing_data/processed_bearing_data.zip"#new data
    url = "https://nextcloud.inovex.de/s/36EYw7zjnYo95sk/download/processed_data.zip"#old data
    path = "/tmp/processed_bearing_data"
    if not os.path.exists(path):
        print("Downloading bearing process data...")
        wget.download(url, '/tmp')
        with zipfile.ZipFile(f"{path}.zip", "r") as zip_ref:
            zip_ref.extractall('/tmp')
        os.remove(f"{path}.zip")
    else:
        print("Bearing process data is omitted because it already exists...")
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
