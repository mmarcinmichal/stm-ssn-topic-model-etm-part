"""
Factory dataset loader
"""
import os
from typing import Optional, Type

from model.datasets.loaders.ag_loader import AGLoader
from model.datasets.loaders.base_data_loader import BaseDataLoader
from model.datasets.loaders.bbc_loader import BBCLoader
from model.datasets.loaders.news_loader import NewsDataLoader
from model.preprocessing.data_set_clustering import DataSetFullImpl
from model.preprocessing.dataset_interface import DatasetInterface


class DatasetLoader:
    """
    Factory class for creating instances of BaseDataLoader subclasses.

    This factory class provides a convenient way to create instances of concrete subclasses
    of BaseDataLoader based on the specified data_set_name.

    Attributes:
        _loader_mapping (dict): A dictionary mapping supported data_set_name types to their
            corresponding dataset loader classes.
    """

    _loader_mapping = {
        "20news": NewsDataLoader,
        "bbc": BBCLoader,
        "ag": AGLoader,
    }

    def __init__(self, name: str, features_limit: int, folder: str):
        self.path = folder
        self.features_limit = features_limit
        self.name = name

    def load_dataset(self) -> DatasetInterface:
        """
        Load or preprocess and save a dataset based on the provided configuration.
        If the preprocessed dataset doesn't exist, this method preprocesses the dataset
        using the configured parameters and saves it. If the dataset already exists, it is
        loaded and returned.

        Returns:
            DatasetInterface: A dataset instance containing preprocessed data.
        Note:
            The `self.name` and `self.features_limit` properties are used
        """
        data_set_name = f"{self.name}_{self.features_limit}"
        data_set_path = os.path.join(self.path, data_set_name)
        if not os.path.exists(data_set_path):
            print(
                f"Preprocessing {self.name} dataset limited to {self.features_limit} features"
            )
            loader: BaseDataLoader = self.get_loader(self.name)
            documents_training, labels_training = loader.gat_training_set()
            document_test, labels_test = loader.gat_test_set()
            data_set: DatasetInterface = DataSetFullImpl(self.features_limit)
            data_set.preprocess_data_set(
                documents_training,
                document_test,
                labels_training,
                labels_test,
                loader.categories(),
            )
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            data_set.save(self.path, data_set_name)
            return data_set
        else:
            print(f"Loading {self.name} dataset from {self.path}")
            return DataSetFullImpl.load(self.path, data_set_name)

    @staticmethod
    def get_loader(data_set_name: str) -> BaseDataLoader:
        """
        Get a data loader instance based on the given dataset name.

        Args:
            data_set_name (str): The name of the dataset to load.

        Returns:
            BaseDataLoader: An instance of a data loader for the specified dataset.

        Raises:
            Exception: If the specified dataset name does not have
                    a corresponding loader implementation.
        """
        loader_class: Optional[
            Type[BaseDataLoader]
        ] = DatasetLoader._loader_mapping.get(data_set_name)
        if loader_class:
            return loader_class()

        raise Exception(f"Loader not implemented for {data_set_name}")
