import numpy as np

from fl_models.util.constants import (
    SPECTRA_SHAPE,
    SPECTRA_CSV_NAME,
)
from fl_models.util.dataloader import (
    df_dict_to_df_dataframe,
    read_feature_dfs_as_dict,
)
from fl_models.util.helper_methods import pop_labels
from fl_models.CNNSpectraFeatures import CNNSpectraFeatures
from fl_models.util.metrics import rmse, correlation_coefficient

if __name__ == "__main__":
    # Read training and validation data
    spectra_training_df, spectra_training_labels = df_dict_to_df_dataframe(
        read_feature_dfs_as_dict(data_set_type=SPECTRA_CSV_NAME)
    )
    spectra_validation_df, spectra_validation_labels = df_dict_to_df_dataframe(
        read_feature_dfs_as_dict(data_set_type=SPECTRA_CSV_NAME)
    )

    spectra_training_dfs = spectra_training_df.to_numpy()
    spectra_training_dfs = np.array(
        [row.reshape(SPECTRA_SHAPE) for row in spectra_training_dfs]
    )

    spectra_validation_dfs = spectra_validation_df.to_numpy()
    spectra_validation_dfs = np.array(
        [row.reshape(SPECTRA_SHAPE) for row in spectra_validation_dfs]
    )

    # Train CNN
    cnn_model = CNNSpectraFeatures(name="Time_Frequency_CNN")
    cnn_model.train(
        training_data=spectra_training_df,
        training_labels=spectra_training_labels,
        validation_data=spectra_validation_df,
        validation_labels=spectra_validation_labels,
        epochs=10
    )
    # Read validation data for metrics computation
    metrics_data = read_feature_dfs_as_dict(data_set_type=SPECTRA_CSV_NAME)
    metrics_labels = pop_labels(metrics_data)

    for k, v in metrics_data.items():
        print(k, ":", v.shape)

    metrics_dict = cnn_model.compute_metrics(
        df_dict=metrics_data,
        labels=metrics_labels,
        metrics_list=[rmse, correlation_coefficient],
    )
    print(metrics_dict)
