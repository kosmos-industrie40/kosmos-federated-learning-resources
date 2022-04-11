from enum import Enum
from fl_models.util.constants import (
    SPECTRA_CSV_NAME,
    FEATURES_CSV_NAME,
)


class DataSetType(Enum):
    spectra = SPECTRA_CSV_NAME
    computed = FEATURES_CSV_NAME
