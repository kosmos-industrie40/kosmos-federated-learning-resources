from typing import Sequence, Callable, Dict, Optional, List
import pandas as pd
import numpy as np
import abc
from fl_models.DataSetType import DataSetType

from tensorflow.keras import Model
from tensorflow.keras.callbacks import History


class DegradationModel(abc.ABC):
    @abc.abstractmethod
    def __init__(self, name: str, data_set_type: DataSetType):
        self.prediction_model: Model = None
        self.svr = None
        self.gpr = None
        self.poly_reg = None
        self.trainings_history = None
        self.name = name
        self.data_set_type = data_set_type

    @abc.abstractmethod
    def predict(self, df_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Takes a list of input vectors and returns RUL predictions for that list.
        :param df_dict:
        :return:
        """
        pass

    @abc.abstractmethod
    def train(
        self,
        training_data: pd.DataFrame,
        training_labels: pd.Series,
        validation_data: Optional[pd.DataFrame],
        validation_labels: Optional[pd.Series],
        epochs: int,
        learning_rate: int = None,
    ) -> History:
        """
        Takes training data and labels and trains a model corresponding to the respective class.
        Also sets the prediction model.
        :return: Returns the trainings history
        """
        pass

    def compute_metrics(
        self,
        df_dict: Dict[str, pd.DataFrame],
        labels: Dict[str, pd.Series],
        metrics_list: Sequence[Callable],
    ) -> Dict[str, Dict[str, float]]:
        predictions = self.predict(df_dict=df_dict)
        result = {}
        for key in df_dict.keys():
            result[key] = {
                metric.__name__: metric(labels[key], predictions[key])
                for metric in metrics_list
            }
        return result

    def get_name(self) -> str:
        return self.name

    def get_data_set_type(self) -> DataSetType:
        return self.data_set_type

    def get_parameters(self) -> List[np.array]:
        return self.prediction_model.get_weights()

    def set_parameters(self, new_weights: List[np.array]) -> None:
        self.prediction_model.set_weights(weights=new_weights)
