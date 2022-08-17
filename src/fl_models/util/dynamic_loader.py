"""
Contains factory functions for plug-in style class loading. When importing the files in
the specified folder are checked for a load_class function.
The registry contains the classes (subclasses of the given ABC)

The files are expected to contain a function called
load_class that returns the class.
"""
import importlib.util
import os
import typing


from fl_models.abstract.abstract_dataset import AbstractDataset

from fl_models.abstract.abstract_usecase import FederatedLearningUsecase

from fl_models.abstract.abstract_model import AbstractModel


_USECASE_LOADER = None
_DATASET_LOADER = None
_MODEL_LOADER = None

_USE_CASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "usecases")
_DATASET_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
_MODEL_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class DynamicLoader:
    """
    Loads all subclasses of a given base_abc in the specified folder into a registry.
    Requires the files to contain a load_class function that returns the class (not an instance!).
    """

    _base_abc_type = None

    def __init__(self, directory, base_abc, load_on_init=True) -> None:
        """
        Initializes the folder, base_class and whether the classes should already be loaded
        """
        # Path where the module searches for subclasses of base_abc
        self.dir_path = directory

        self._base_abc = base_abc
        self._base_abc_type = base_abc

        # Dictionary containing the instances
        self._registry: typing.Dict[str, self._base_abc_type] = {}

        if load_on_init:
            self.load()

    def load(self):
        """Load classes from self.dir_path. If the class has a staticmethod 'get_registry_name' the
        return value is used as registry key. Else the __name__ member is used."""

        # Walk sub directories
        for (root, _, files) in os.walk(self.dir_path):

            # Look for files in directory
            for current_file in files:
                if (
                    len(current_file) < 3
                    or not current_file[-3:] == ".py"
                    or current_file == "__init__.py"
                ):
                    # Not a python file -> ignore
                    continue

                # Import class
                try:
                    # Dynamic import
                    full_file_path = os.path.join(root, current_file)
                    module_spec = importlib.util.spec_from_file_location(
                        root, full_file_path
                    )

                    current_module = importlib.util.module_from_spec(module_spec)
                    module_spec.loader.exec_module(current_module)

                    if not hasattr(current_module, "load_class"):
                        # Load class is missing

                        print(
                            f"Error when importing from {full_file_path}: "
                            'Cannot find function "load_class".'
                        )
                        continue

                    try:
                        class_object = current_module.load_class()

                        if not issubclass(class_object, self._base_abc_type):
                            # Wrong interface

                            print(
                                f"Error when importing from {full_file_path}: "
                                + f"Not a subclass of {self._base_abc_type.__name__}."
                            )
                            continue

                        registry_key = class_object.__name__
                        if hasattr(class_object, "get_registry_name"):
                            registry_key = class_object.get_registry_name()

                        self._registry[registry_key] = class_object
                    except TypeError as type_error:
                        print(
                            f"Error when importing from {full_file_path}: {type_error}."
                        )

                except ImportError as import_error:
                    print(
                        f"Error when importing from {full_file_path}: {import_error.msg}"
                    )

            return  # Return after first iteration to avoid walking subdirectories

    def refresh(self):
        """Resets registry and walks subdirectories again"""
        self._registry = {}
        self.load()

    def get(self, key: str):
        """Factory method for classes. Uses the given key to find the \
            matching class


        Example:

            loader = DynamicLoader("dir", TestABC)
            class_name = loader.get("MyClass")

            instance = class_name(parameters)

        :param key: Key used to find the correct instance. If the class has a staticmethod
                'get_registry_name' the return value is used as key. Else the __name__ member is
                used.
        :type key: str
        :raises ValueError: Raised when the requested class was not found
        :return: Requested class.
        :rtype: Subclass object (not instance!) of the given abstract class
        """
        try:
            return self._registry[key]

        except KeyError as unknown_class:
            raise ValueError(f"Unknown class name: {key}") from unknown_class

    def get_names(self) -> typing.List[str]:
        """
        Returns a list of all available classes.

        :return: List of all available classes
        :rtype: List[str]
        """
        return list(self._registry.keys())


def load_dataset(dataset_name: str, *args, **kwargs) -> AbstractDataset:
    """
    Returns instance of specified dataset. Raises a ValueError if the dataset could not be found
    :param dataset_name: Name of the dataset class

    :return: Instance of the specified dataset, a child of AbstractDataset
    """
    # pylint: disable=global-statement
    # Only load when necessary to avoid overhead
    global _DATASET_LOADER

    if _DATASET_LOADER is None:
        # Init
        _DATASET_LOADER = DynamicLoader(_DATASET_FOLDER, AbstractDataset)
    return _DATASET_LOADER.get(dataset_name)(*args, **kwargs)


def load_usecase(usecase_name, *args, **kwargs) -> FederatedLearningUsecase:
    """
    Returns instance of specified usecase. Raises a ValueError if the usecase could not be found
    :param usecase_name: Name of the usecase class

    :return: Instance of the specified usecase, a child of FederatedLearningUsecase
    """
    # pylint: disable=global-statement
    # Only load when necessary to avoid overhead
    global _USECASE_LOADER

    if _USECASE_LOADER is None:
        # INIT
        _USECASE_LOADER = DynamicLoader(_USE_CASE_PATH, FederatedLearningUsecase)

    return _USECASE_LOADER.get(usecase_name)(*args, **kwargs)


def load_model(model_name: str, *args, **kwargs) -> AbstractModel:
    """
    Returns instance of specified model. Raises a ValueError if the model could not be found
    :param model_name: Name of the model class

    :return: Instance of the specified model, a child of AbstractModel
    """
    # pylint: disable=global-statement
    # Only load when necessary to avoid overhead
    global _MODEL_LOADER

    if _MODEL_LOADER is None:
        # INIT
        _MODEL_LOADER = DynamicLoader(_MODEL_FOLDER, AbstractModel)
    return _MODEL_LOADER.get(model_name)(*args, **kwargs)
