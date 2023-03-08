import os
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from seqlbtoolkit.data import merge_list_of_lists
from seqlbtoolkit.training.dataset import (
    BaseDataset,
    DataInstance,
    feature_lists_to_instance_list,
)

logger = logging.getLogger(__name__)


class Dataset(BaseDataset):
    def __init__(self):
        super().__init__()

        self._features = None
        self._smiles = None
        self._lbs = None
    
    @property
    def features(self):
        return self._features
    
    @property
    def lbs(self):
        return self._lbs

    def prepare(self, config, partition):
        """
        Prepare dataset for training and test

        Parameters
        ----------
        config: configurations
        partition: dataset partition; in [train, valid, test]

        Returns
        -------
        self
        """

        assert partition in ['train', 'valid', 'test'], \
            ValueError(f"Argument `partition` should be one of 'train', 'valid' or 'test'!")

        file_path = os.path.normpath(os.path.join(config.data_dir, f"{partition}.pt"))

        if file_path and os.path.exists(file_path):
            self.load(file_path)
        else:
            raise FileNotFoundError(f"File {file_path} does not exist!")

        self.data_instances = feature_lists_to_instance_list(
            DataInstance,
            features=self._features, lbs=self._lbs
        )

        return self

    def load(self, file_path: str):
        """
        Load data
        """

        data_dict = torch.load(file_path)

        try:
            self._features = [torch.from_numpy(features).to(torch.float) for features in data_dict["features"]]
            self._smiles = data_dict["smiles"]
        except KeyError as exception:
            logger.error("The target file is not compatible! "
                         "It must have `features` and `smiles` as keys.")
            raise exception

        if isinstance(data_dict["targets"][0], (tuple, list)):
            self._lbs = merge_list_of_lists(data_dict["targets"])
        else:
            self._lbs = data_dict["targets"]

        return self
