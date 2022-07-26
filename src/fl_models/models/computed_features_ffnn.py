"""
Contains a FFNN model for RUL prediction
"""
from typing import Sequence, Optional, Tuple
import pandas as pd
from tensorflow.keras.callbacks import History

from fl_models.ffnn import fit_ffnn, build_ffnn_model
from fl_models.util.constants import ALL_FEATURES
from fl_models.abstract.abstract_model import AbstractModel
from fl_models.data_set_type import DataSetType


class ComputedFeaturesFFNN(AbstractModel):
    """
    Computes rul using precomputed features with a FFNN
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        feature_list: Sequence[str] = None,
        dropout: bool = True,
        num_hidden_layers: int = 2,
        num_hidden_units: int = 512,
        learning_rate: float = None,
    ):
        super().__init__(learning_rate=learning_rate)

        if feature_list is None:
            feature_list = ALL_FEATURES.copy()

        input_shape = 2 * len(feature_list)
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_units = num_hidden_units
        self.prediction_model = build_ffnn_model(
            input_shape,
            self.dropout,
            hidden_layers=self.num_hidden_layers,
            hidden_units=self.num_hidden_units,
        )
        self.feature_list = [
            prefix + f for f in feature_list for prefix in ["h_", "v_"]
        ]

    def train(
        self,
        training_data: pd.DataFrame,
        training_labels: pd.Series,
        epochs: int,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> History:
        """
        Takes training data and labels and trains a model corresponding to the respective class.
        Also sets the prediction model.

        :param training_data: DataFrame or array of features. First axis is interpreted as number of
            samples
        :type training_data: pd.DataFrame
        :param training_labels: Series of labels. Has to be the same length as training_data
        :type training_labels: pd.Series
        :param validation_data: Tuple containing (validation features, validation labels). \
            DataFrame or array of features used for validation. First axis is \
            interpreted as number of samples
        :type validation_data: Tuple[pd.DataFrame, pd.Series], optional
        :param epochs: Number of training epochs
        :type epochs: int
        :return: Returns the trainings history. Is expected to contain a list of losses under
            history.history["loss"]
        :rtype: History
        """
        assert isinstance(
            training_data, pd.DataFrame
        ), f"'training_data' has wrong type: {type(training_data)}! Supported type(s): DataFrame"
        assert isinstance(
            training_labels, pd.Series
        ), f"'training_labels' has wrong type: {type(training_labels)}! Supported type(s): Series"
        training_data = training_data[self.feature_list]
        if validation_data is not None:
            assert isinstance(validation_data[0], pd.DataFrame), (
                f"'validation_data' has wrong type: {type(validation_data[0])}! "
                + "Supported type(s): DataFrame"
            )
            assert isinstance(validation_data[1], pd.Series), (
                f"'validation_labels' has wrong type: {type(validation_data[1])}! "
                + "Supported type(s): Series"
            )
            validation_data = (
                validation_data[0][self.feature_list],
                validation_data[1],
            )

        # pylint: disable=attribute-defined-outside-init
        # defined in super
        self.prediction_model, self.trainings_history = fit_ffnn(
            ffnn=self.prediction_model,
            training_data=training_data,
            training_labels=training_labels,
            epochs=epochs,
            validation_data=validation_data,
            learning_rate=self.learning_rate,
        )
        return self.trainings_history

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Takes a dataframe as input and performs a prediction on it.
        The first axis is interpreted as number of samples

        :param data: DataFrame or array that contains the features for prediction
        :type data: Union[pd.DataFrame, np.ndarray]
        :return: Predictions, has the length data.shape[0]
        :rtype: pd.Series
        """
        assert isinstance(
            data, pd.DataFrame
        ), f"'data' has wrong type: {type(data)}! Supported type(s): DataFrame"
        return pd.Series(
            self.prediction_model.predict(data[self.feature_list])[:, 0],
            name="rul_predictions",
        )

    def __str__(self) -> str:
        """
        Overwrites default string representation

        :return: String containing model features
        :rtype: str
        """
        return "Features: " + str(self.feature_list) + " FFNN rul prediction."

    @staticmethod
    def get_data_set_type() -> DataSetType:
        """
        Getter for the dataset type

        :return: Computed dataset type
        :rtype: DataSetType
        """
        return DataSetType.computed


def load_class():
    """Returns the class in this file

    Returns:
        model_abc.AbstractModel
    """
    return ComputedFeaturesFFNN
