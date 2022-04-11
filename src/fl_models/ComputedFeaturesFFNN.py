from typing import Dict, Sequence, Optional
import pandas as pd
from tensorflow.keras.callbacks import History

from fl_models.util.helper_methods import flatten_predictions
from fl_models.ffnn import fit_ffnn
from fl_models.DegradationModel import DegradationModel
from fl_models.DataSetType import DataSetType


class ComputedFeaturesFFNN(DegradationModel):
    def __init__(self, feature_list: Sequence[str], name: str):
        DegradationModel.__init__(self, name, DataSetType.computed)
        self.feature_list = [
            prefix + f for f in feature_list for prefix in ["h_", "v_"]
        ]

    def train(
        self,
        training_data: pd.DataFrame,
        training_labels: pd.Series,
        validation_data: Optional[pd.DataFrame],
        validation_labels: Optional[pd.Series],
        epochs: int,
    ) -> History:

        training_data = training_data[self.feature_list]
        if validation_data is not None:
            validation_data = validation_data[self.feature_list]
        self.prediction_model, self.trainings_history = fit_ffnn(
            x=training_data,
            y=training_labels,
            epochs=epochs,
            dropout=True,
            validation_data=(validation_data, validation_labels),
        )
        return self.trainings_history

    def predict(self, df_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        result = {}
        for bearing, df in df_dict.items():
            assert "RUL" not in df.keys()
            result[bearing] = pd.Series(
                flatten_predictions(
                    self.prediction_model.predict(df[self.feature_list])
                ),
                name="rul_predictions",
            )
        return result

    def __str__(self):
        return "Features: " + str(self.feature_list) + " FFNN rul prediction."
