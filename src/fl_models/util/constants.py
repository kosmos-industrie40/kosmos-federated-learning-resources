SPECTRA_SHAPE = (129, 21, 2)

# File names
FEATURES_CSV_NAME = "features.csv.gz"
SPECTRA_CSV_NAME = "spectra.csv.gz"
RAW_CSV_NAME = "acc.csv"

# Metric keys
RMSE_KEY = "RMSE"
CORR_COEFF_KEY = "PCC"
STANDARD_DEVIATION_KEY = "STD"

BASIC_STATISTICAL_FEATURES = [
    "mean",
    "maximum",
    "minimum",
    "root_mean_square",
    "peak_to_peak_value",
    "skewness",
    "kurtosis",
    "variance",
    "peak_factor",
    "change_coefficient",
    "clearance_factor",
    "abs_energy",
]

ENTROPY_FEATURES = ["shannon_entropy", "permutation_entropy", "spectral_entropy"]

FREQUENCY_FEATURES = [
    "frequency_mean",
    "frequency_maximum",
    "frequency_minimum",
    "frequency_root_mean_square",
    "frequency_peak_to_peak_value",
    "frequency_skewness",
    "frequency_kurtosis",
    "frequency_variance",
    "frequency_peak_factor",
    "frequency_change_coefficient",
    "frequency_clearance_factor",
    "frequency_abs_energy",
]

ALL_FEATURES = BASIC_STATISTICAL_FEATURES + ENTROPY_FEATURES + FREQUENCY_FEATURES
