from typing import Optional
from fl_models.util.constants import ALL_FEATURES
from fl_models.ModelType import ModelType
from fl_models.ComputedFeaturesFFNN import ComputedFeaturesFFNN
from fl_models.CNNSpectraFeatures import CNNSpectraFeatures
from fl_models.DegradationModel import DegradationModel


def get_degradation_model(
    model_name: str, model_type: ModelType
) -> Optional[DegradationModel]:
    if model_type == ModelType.COMPUTED_FEATURES_FFNN:
        return ComputedFeaturesFFNN(name=model_name, feature_list=ALL_FEATURES)
    elif model_type == ModelType.SPECTRA_CNN:
        return CNNSpectraFeatures(name=model_name)
    else:
        return None
