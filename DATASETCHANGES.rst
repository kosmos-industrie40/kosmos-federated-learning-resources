===============
BEARING DATASET
===============

The dataset utilized in the bearing usecase is a modified version of the FEMTO-ST Bearings Dataset [NECT2012]_. This file contains further information on the modifications applied.
The dataset is downloaded from `here <https://artifactory.inovex.de/artifactory/KOSMos-public/processed_bearing_data.zip>`_.

Changes
=======

- Temperature measurements were dropped
- For each bearing, all vibration measurement files (``acc\_*.csv``) were combined into two files (*spectra.csv* and *features.csv*; each ``acc\_*.csv`` file represents one row)
- **spectra.csv**: Spectral transformation of horizontal and vertical vibration using a short term fourier transformation + RUL.
- **features.csv**: Features calculated regarding horizontal and vertical vibration: 

  - mean
  - maximum
  - minimum
  - root_mean_square
  - peak_to_peak_value
  - skewness
  - kurtosis
  - variance
  - peak_factor
  - change_coefficient
  - clearance_factor
  - abs_energy
  - shannon_entropy
  - permutation_entropy
  - spectral_entropy
  - frequency_mean
  - frequency_maximum
  - frequency_minimum
  - frequency_root_mean_square
  - frequency_peak_to_peak_value
  - frequency_skewness
  - frequency_kurtosis
  - frequency_variance
  - frequency_peak_factor
  - frequency_change_coefficient
  - frequency_clearance_factor
  - frequency_abs_energy
  - RUL


References
==========

.. [NECT2012] Patrick Nectoux, Rafael Gouriveau, Kamal Medjaher, Emmanuel Ramasso, Brigitte Morello, Noureddine Zerhouni, Christophe Varnier. (2012). PRONOSTIA : An experimental platform for bearings accelerated degradation tests. In IEEE International Conference on Prognostics and Health Management (pp. 1–8). Denver.
