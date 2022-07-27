=================
Developer's Guide
=================

This guide describes how to implement a new usecase with a new dataset and model.

Table of Contents:
==================

1. Introduction
2. Implementation
3. Other Files

Introduction
============

Prerequisites
-------------

Dynamic Loading
---------------

This project uses a dynamic loading system for models, datasets, usecases and clients. 
This way, classes can be loaded using their class name. Currently, all possible class files have to
be located in the same folder and inherit from the same interface. The loader is then given the
interface class and the path to the folder loading all suitable classes into its registry. 

.. list-table:: Locations the different class types are loaded from.
   :widths: 25 25
   :header-rows: 1

   * - Class Type
     - Path
   * - Models
     - fl_models/src/fl_models/models
   * - Datasets
     - fl_models/src/fl_models/datasets
   * - Usecases
     - fl_models/src/fl_models/usecases
   * - Clients
     - fl_client/src/fl_client/clients

In order for this mechanism to work, the class file has to have a certain structure:

* The class in the file has to implement the given interface
* The file needs to contain a function called :code:`load_class` that returns the class.
* (optional) The registry key is usually the :code:`__name__` attribute. This value can be changed by implementing a method in the class called :code:`get_registry_name`, that returns the custom value.

A short example of a dynamically loaded class:

.. code-block:: python

    class MyDynamicClass(Interface):
        def interfaceMethod(self, ...):
            ....
        
        @staticmethod
        def get_registry_name():
            return "MyCustomName"
    
    def load_class():
        return MyDynamicClass


Information Flow during Runtime
-------------------------------



Implementation
==============

This section focusses on the implementation of the different abstract base classes in the project.
The abstract base classes are located in :code:`fl_models.abstract`. In this short example we create a 
new use case called :code:`MyUsecase` that uses the dataset :code:`MyDataset` and the model :code:`MyModel`. 
The full files can be found in `examples <examples/>`_.

Dataset
-------

The class :code:`AbstractDataset` in :code:`fl_models.abstract.abstract_dataset` defines the interface for loading a new dataset.
The interface has only one public method, :code:`get_dataframe`, which returns the loaded dataframe. 

1. In order to create the new dataset we have to create :code:`my_dataset.py` in the folder `fl_models/datasets/ <src/fl_models/datasets>`_.
2. Next we have to create a subclass of :code:`AbstractDataset`:

.. code-block:: python

  import json
  import pandas as pd
  from fl_models.abstract.abstract_dataset import AbstractDataset

  class MyDataset(AbstractDataset):

    pass

3. Our dataset loads a dictionary from a string as pandas DataFrame. We want the user to be able to specify this dictionary, therefore
we have to add it as a parameter in the :code:`__init__` function. Note: Only use keyword-arguments!

.. code-block:: python

  def __init__(self, user_dict = None):
    # Since we can only use keyword arguments, we have to use assertion in order to assure the existence
    assert user_dict is not None, "Please specify a dictionary that should be loaded!"

    self.user_dict = json.loads(user_dict)

4. As a last step we have to read the DataFrame and return it in :code:`get_dataframe` so it can be used by the usecase.

.. code-block:: python

  def get_dataframe(self):
    return pd.DataFrame.from_dict(self.user_dict)

5. As a last step we have to add the :code:`load_class` function to the file so the dynamic loading system can find the class.

.. code-block:: python

  def load_class():
    return MyDataset

The full file can be found at `examples/my_dataset.py <examples/my_dataset.py>`_.


Model
-----

The :code:`AbstractModel` class in :code:`fl_models.abstract.abstract_model` defines the interface for tensorflow models.

.. list-table:: Interfaces defined in `AbstractModel`.
   :widths: 25 25 25
   :header-rows: 1

   * - Name
     - Description
     - When to overwrite
   * - :code:`__init__`
     - Gets the learning_rate as input and defines the expected attributes: :code:`self.learning_rate`, :code:`self.prediction_model` (a tf.Model) and :code:`self.tranings_history`.
     - Most of the time since :code:`self.prediction_model` and :code:`self.tranings_history` default to None. Should always be called by subclasses.
   * - :code:`predict`
     - Gets a pandas DataFrame or a numpy array as input and uses them as input for the inference. Returns either a pandas Series or a numpy array of predictions. The input types can be more specific for subclasses.
     - Always!
   * - :code:`train`
     - Gets the features, labels, number of epochs and optionally a tuple of validation data (features, labels) as input. Fits the model using the input data and returns the tensorflow training history.
     - Always!
   * - :code:`compute_metrics`
     - Gets called by the usecase to evaluate the current model. Gets the features, labels, a list of metrics and optionally a callback function for logging. Calls the :code:`predict` function and calculates the metrics on the results.
     - Only for very specific models since the definition should work with most usecases and model definitions.
   * - :code:`get_weights`
     - Getter for the model weights.
     - Definition should work with most standard models.
   * - :code:`set_weights`
     - Setter for the model weights.
     - Definition should work with most standard models.

We want to create a simple model with one hidden layer.

1. In order to create a new model we have to create :code:`my_model.py` in the folder `fl_models/models/ <src/fl_models/models>`_.
The skeleton of the file is similar to :code:`my_dataset.py`:

.. code-block:: python

  import numpy as np
  import tensorflow as tf
  from fl_models.abstract.abstract_model import AbstractModel

  class MyModel(AbstractModel):

    def __init__(self, input_shape, learning_rate=None):
      pass

    def predict(self, data):
      pass
    
    def train(self, training_data, training_labels, epochs, validation_data = None):
      pass

  def load_class():
    return MyModel

2. First we have to initialize our model (:code:`self.prediction_model`). Since the dataset is based on user input,
the feature shape has to be given to the model during initialization.

.. code-block:: python

  def __init__(self, input_shape, learning_rate=None):
    super().__init__(learning_rate=learning_rate)

    self.prediction_model = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(input_shape)),
      tf.keras.layers.Dense(4),
      tf.keras.layers.Dense(1),
    ])

3. Next we have to implement the :code:`predict` function. It will support both numpy arrays and dataframes.

.. code-block:: python

  def predict(self, data):
    features = data if isinstance(data, np.ndarray) else data.to_numpy()

    return self.prediction_model.predict(features)

4. As a last step we have to implement the :code:`train` function. The full file can be found at `examples/my_model.py <examples/my_model.py>`_

.. code-block:: python

  def train(self, training_data, training_labels, epochs, validation_data=None):

    val_tuple = validation_data
    if validation_data is not None:
      # Convert to numpy
      val_tuple = (
        val_tuple[0] if isinstance(val_tuple[0], np.ndarray) else val_tuple[0].to_numpy(),
        val_tuple[1] if isinstance(val_tuple[1], np.ndarray) else val_tuple[1].to_numpy(),
      )

    # Set optimizer and loss
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=self.learning_rate if self.learning_rate is not None else 0.2
    )

    self.prediction_model.compile(loss="mse", optimizer=optimizer)

    # Train model
    return self.prediction_model.fit(
        training_data if isinstance(training_data, np.ndarray) else training_data.to_numpy(),
        training_labels if isinstance(training_labels, np.ndarray) else training_labels.to_numpy(),
        epochs=epochs,
        validation_data=val_tuple,
        verbose=1,
    )

Usecase
-------

The usecase handles the information flow during the federated learning process, defines which model and dataset should be used and 
is the direct interface to `flwr <https://flower.dev/docs/index.html>`_.

.. list-table:: Interfaces defined in :code:`FederatedLearningUsecase`.
   :widths: 25 25 25
   :header-rows: 1

   * - Name
     - Description
     - When to overwrite
   * - :code:`__init__`
     - Gets the log_mlflow flag as input and defines expected attributes. A list of attributes can be found below.
     - Most of the time since the attributes default to None. Should always be called by subclasses.
   * - :code:`get_model_name`
     - Static function that returns the model associated with the usecase.
     - Always!
   * - :code:`get_dataset_name`
     - Static function that returns the dataset associated with the usecase.
     - Always!
   * - :code:`get_data`
     - Getter for the (feature) data. The resulting data is used as input for :code:`eval_fn`.
     - Always, since the default implementation ignores the :code:`flat` parameter.
   * - :code:`get_labels`
     - Getter for the labels. The resulting labels is used as input for :code:`eval_fn`.
     - Always, since the default implementation ignores the :code:`flat` parameter.
   * - :code:`get_identifiers`
     - Getter for the data identifiers. These are used for grouping and logging in :code:`eval_fn`.
     - Definition should work with most usecases.
   * - :code:`get_number_of_samples`
     - Getter for the number of data rows in the dataset.
     - Definition should work with most standard models.
   * - :code:`eval_fn`
     - [flwr interface] Evaluates the model with the given weights.
     - Definition should work with most standard models.
   * - :code:`get_model`
     - Getter for the model instance.
     - Definition should work with most standard models.
   * - :code:`get_dataset`
     - Getter for the dataset instance.
     - Definition should work with most standard models.

.. list-table:: Attributes defined in :code:`FederatedLearningUsecase`.
   :widths: 25 25
   :header-rows: 1

   * - Name
     - Description
   * - :code:`current_fed_rnd`
     - Counter for the federated learning rounds.
   * - :code:`metrics`
     - List of metrics (callables) that will be used to evaluate the model.
   * - :code:`abstract_model`
     - Instance of a subclass of :code:`AbstractModel`.
   * - :code:`dataset`
     - Instance of a subclass of :code:`AbstractDataset`
   * - :code:`log_mlflow`
     - Flag that indicates whether or not the metrics should be logged to mlflow.
   * - :code:`data`
     - Feature data that will be used for training/evaluating the model.
   * - :code:`labels`
     - Labels that will be used for training/evaluating the model.
   * - :code:`identifiers`
     - Data identifiers used for logging and grouping the data.

We are implementing a usecase that reads a dictionary from the configuration file and performs a prediction on it.

1. In order to create a new usecase we have to create :code:`my_usecase.py` in the folder `fl_models/usecases/ <src/fl_models/usecases>`_.
The skeleton of the file is similar to :code:`my_dataset.py`:

.. code-block:: python

  import numpy as np
  import pandas as pd

  from fl_models.abstract.abstract_model import AbstractModel
  from fl_models.abstract.abstract_dataset import AbstractDataset
  from fl_models.util.metrics import rmse

  from fl_models.abstract.abstract_usecase import FederatedLearningUsecase
  from fl_models.util.dynamic_loader import load_model, load_dataset


  class MyUsecase(FederatedLearningUsecase):

    def __init__(
      self, user_dict = None, target_column = None, identifier_column = None,
      log_mlflow = True, learning_rate = None, **kwargs,
    ):
      pass

    @staticmethod
    def get_model_name():
      pass

    @staticmethod
    def get_dataset_name():
      pass

    def get_data(self, flat = False):
      pass

    def get_labels(self, flat = False):
      pass
          
  def load_class():
    return MyUsecase

2. Now we want to define which datasets and models are used in the usecase. This is is done by defining the static functions :code:`get_model_name` and :code:`get_dataset_name`.
We want to use our previously defined model :code:`MyModel` and our custom dataset :code:`MyDataset`:

.. code-block:: python
  
  @staticmethod
  def get_model_name():
    return "MyModel"

  @staticmethod
  def get_dataset_name():
    return "MyDataset"

3. In the :code:`__init__` we want to load the model, dataset and structure our data. Apart from the user-defined dictionary
we need the name of the target/label column and the name of the identifier column. The identifier column is used to group the data.

.. code-block:: python

  def __init__(
    self, user_dict = None, target_column = None, identifier_column = None,
    log_mlflow = True, learning_rate = None, **kwargs,
  ):
    assert user_dict is not None, "Missing 'user_dict' setting in configuration."
    assert target_column is not None, "Missing 'target_column' setting in configuration."
    assert identifier_column is not None, "Missing 'identifier_column' setting in configuration."

    super().__init__(log_mlflow=log_mlflow)

    self.metrics = [rmse]

    # Initialize Dataset
    # -> parameters are the name of the dataset and the rest are the parameters of "MyDataset"
    self.dataset: AbstractDataset = load_dataset(
      self.get_dataset_name(), user_dict=user_dict
    )

    self.ident_col = identifier_column
    self.target_column = target_column

    # Load and split dataset
    full_df = self.dataset.get_dataframe()
    self.data: pd.DataFrame = full_df.drop(columns=[self.target_column])
    self.labels: pd.DataFrame = full_df[self.ident_col, self.target_column]
    self.identifiers: pd.Series = full_df[self.ident_col].unique()

    # Load model (needs input shape)
    # -> parameters are the name of the model and the rest are the parameters of "MyModel"
    self.abstract_model: AbstractModel = load_model(
      self.get_model_name(),
      learning_rate=learning_rate,
      input_shape=self.data.shape[1] - 1,  # -1 Since we still have the identifier column
    )

    # Warn in case there are unused arguments
    self.check_for_unused_args(kwargs)

4. As a last step we have to update the getters for :code:`data` and :code:`labels` so it supports our
dataset and grouped data. You can find the full file at `examples/my_usecase.py <examples/my_usecase.py>`_

.. code-block:: python
  def get_data(self, flat = False):
    if flat:
      return self.data.drop(columns=[self.ident_col])

    grouped_data = self.data.groupby(self.ident_col, as_index=False)
    return [
      grouped_data.get_group(val).drop(columns=[self.ident_col]) for val in self.identifiers
    ]

  def get_labels(self, flat = False):
    if flat:
      return self.labels[self.target_column]

    grouped_labels = self.labels.groupby(self.ident_col, as_index=False)
    return [
      grouped_labels.get_group(val)[self.target_column] for val in self.identifiers
    ]

Other Files
===========

As a last step the configuration files have to be updated so our usecase gets triggered. Additionally, for some very special usecases the client also has to be updated.

Updating the Configuration Files
----------------------------

There are two configuration files that have to be updated. One on the client side at `fl_client/src/fl_client/config.yaml <https://github.com/kosmos-industrie40/kosmos-federated-learning-client/blob/release/src/fl_client/config.yaml>`_
and one on the server side at `fl_server/src/fl_server/config.yaml <https://github.com/kosmos-industrie40/kosmos-federated-learning-server/blob/release/src/fl_server/config.yaml>`_.

In the server configuration file the :code:`usecase` section has to be updated:

.. code-block:: yaml

  usecase:
    name: "MyUsecase"  # Set the usecase name
    params:  # Parameters (kwargs) that are only applied on the server side
      user_dict: 
        "{\"first_col\": [\"A\", \"A\", \"B\", \"B\"], \"second_col\": [0.1, 0.2, 0.3, 0.4], \"third_col\": [0.11, 0.22, 0.33, 0.44], \"fourth_col\": [5, 6, 7, 8]}"
    broadcast:  # Parameters (kwargs) that are applied on both the server and client side.
      target_column: "fourth_col"
      identifier_column: "first_col"

The client configuration can hold settings for multiple usecases. The new setting can just be appended to it:

.. code-block:: yaml

  usecase:
    BearingUseCase:
      ...
    TurbofanUseCase:
      ...
    MyUsecase:
      0:  # Settings can be different for each client
        user_dict: 
          "{\"first_col\": [\"A\", \"A\", \"B\", \"B\"], \"second_col\": [0.5, 0.6, 0.7, 0.8], \"third_col\": [0.55, 0.66, 0.77, 0.88], \"fourth_col\": [9, 10, 11, 12]}"]
      1:  # Settings can be different for each client
        user_dict: 
          "{\"first_col\": [\"A\", \"A\", \"B\", \"B\"], \"second_col\": [0.5, 0.6, 0.7, 0.8], \"third_col\": [0.55, 0.66, 0.77, 0.88], \"fourth_col\": [9, 10, 11, 12]}"
      2:  # Settings can be different for each client
        user_dict: 
          "{\"first_col\": [\"A\", \"A\", \"B\", \"B\"], \"second_col\": [0.5, 0.6, 0.7, 0.8], \"third_col\": [0.55, 0.66, 0.77, 0.88], \"fourth_col\": [9, 10, 11, 12]}"


Updating the Client
-------------------

The client takes care of the training process and is a subclass of the `flwr.client.NumPyClient`.
The folder `fl_client\src\fl_client\clients` is checked for client implementations that can be loaded dynamically.
Even though dynamic loading is supported, the pre-implemented :code:`BasicClient` should support most usecases. 
