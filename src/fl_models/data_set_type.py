"""
Module defining different dataset types for bearing dataset
"""
from enum import Enum
from fl_models.util.constants import (
    SPECTRA_CSV_NAME,
    FEATURES_CSV_NAME,
)

# pylint: disable=invalid-name
# Enum name
class DataSetType(Enum):
    """
    Enum for different dataset types in bearing dataset
    """

    spectra = SPECTRA_CSV_NAME
    computed = FEATURES_CSV_NAME
