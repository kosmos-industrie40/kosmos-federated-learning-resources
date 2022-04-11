from typing import Dict, Optional
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import History

from fl_models.util.constants import SPECTRA_SHAPE
from fl_models.util.helper_methods import flatten_predictions

from fl_models.cnn_model import fit_cnn, build_cnn
from fl_models.DegradationModel import DegradationModel
from fl_models.DataSetType import DataSetType


class CNNSpectraFeatures(DegradationModel):
    def __init__(self, name):
        DegradationModel.__init__(self, name, DataSetType.spectra)
        self.prediction_model = build_cnn()

    def train(
        self,
        training_data: pd.DataFrame,
        training_labels: pd.Series,
        validation_data: Optional[pd.DataFrame],
        validation_labels: Optional[pd.Series],
        epochs: int,
        learning_rate: float = None,
    ) -> History:
        spectra_dfs = training_data.to_numpy()
        spectra_dfs = np.array([row.reshape(SPECTRA_SHAPE) for row in spectra_dfs])
        if validation_data is not None and validation_labels is not None:
            validation_data = validation_data.to_numpy()
            validation_data = np.array(
                [row.reshape(SPECTRA_SHAPE) for row in validation_data]
            )

        self.prediction_model, self.trainings_history = fit_cnn(
            spectra_dfs,
            training_labels,
            epochs=epochs,
            input_shape=SPECTRA_SHAPE,
            validation_data=(validation_data, validation_labels),
            learning_rate=learning_rate,
        )

        return self.trainings_history

    def predict(self, df_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        df_dict = _reformat_spectra(df_dict)
        result = {}
        for bearing, df in df_dict.items():
            result[bearing] = pd.Series(
                flatten_predictions(self.prediction_model.predict(df)),
                name="rul_predictions",
            )
        return result


def _reformat_spectra(df_dict: Dict[str, pd.DataFrame]):
    return {
        bearing: df.to_numpy().reshape(
            (df.shape[0], SPECTRA_SHAPE[0], SPECTRA_SHAPE[1], SPECTRA_SHAPE[2])
        )
        for bearing, df in df_dict.items()
    }
