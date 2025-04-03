"""
This module provides functionality for preparing and handling EGFET (Extended-Gate Field-Effect Transistor) dataset for machine learning tasks. It includes functions and classes to:

- Prepare data by organizing it, extracting labels, concentrations, and stratification groups.
- Split the dataset into training and testing sets while considering stratification groups.
- Define a custom PyTorch Dataset class (`EGFETDataset`) for loading and accessing the EGFET data.
- Build train and test dataset splits from the given data path.
- Yield dataset folds for cross-validation using repeated group K-fold splitting.

Functions:
- prepare_data(data, stratify_level=2): Organizes the data and extracts labels, concentrations, and stratification groups.
- num_unique_groups(stratify_groups): Returns the number of unique stratification groups.
- get_train_test_group_splits(stratify_groups, test_size=0.2): Splits the dataset into training and testing sets based on stratification groups.

Classes:
- EGFETDataset: A custom PyTorch Dataset class for EGFET data.
    - __init__(self, path, exclude_experiment, exclude_concentration, include_only, downsample, verbose_loading): Initializes the dataset by loading and preparing the data.
    - __len__(self): Returns the length of the dataset.
    - __getitem__(self, idx): Returns the data and target at the specified index.
    - build_dataset_splits(cls, path, exclude_experiment, exclude_concentration, include_only, downsample, verbose_loading, test_size): Builds train and test dataset splits.
    - yield_dataset_folds(cls, path, exclude_experiment, exclude_concentration, include_only, downsample, verbose_loading, n_splits, n_repeats): Yields train and test dataset folds for cross-validation.
"""

# System imports
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pytorch_lightning as pl
import torch

from pathlib import Path
from typing import List, Optional

# My libraries
from histones.egfet.load_data import load_egfet_dataset
from pipelinegen_histones.utils import RepeatedGroupKFold
from pipelinegen.core.data.factory import AbstractDatasetBuilder
from pipelinegen.core.data.utils import MultiArgCompose

def prepare_data(data, stratify_level=2):
    """Get the labels, concentrations, and split groups for the dataset

    Args:
        data (pd.DataFrame): The data to be prepared
        stratify_level (int): The level at which to stratify the data.
            The higher the more strict the stratification. Accepts values
            between0 and 3 (default: 2)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The organized data,
            the analyte type, the concentration, and the stratify groups
    """
    organized_data = []
    analyte_type = []
    concentration = []
    lengths = []
    stratifykey = 0
    stratify_map = dict()
    # get indices
    indices = data.index
    # drop last index
    indices = indices.droplevel(-1).unique()
    stratify_groups = []
    min_length = 100000
    for index in indices:
        idx = index
        index_data = data.loc[idx]

        if len(index_data) < 60:
            continue
        stratify_name = "".join([str(i) for i in idx[:-stratify_level]])
        key = stratify_map.get(stratify_name, stratifykey := stratifykey + 1)
        stratify_map[stratify_name] = key
        stratify_groups.append(key)
        lengths.append(len(index_data))
        min_length = min(min_length, len(index_data))
        concentration.append(idx[1])
        organized_data.append(index_data["Drain Current (nA)"].values[:60])
        if (
            ("Hapt" in index[0] or "H4" in index[0])
            and "CTH" not in index[0]
            and "BSA" not in index[0]
        ):
            analyte_type.append(0)
        if "BSA" in index[0]:
            analyte_type.append(1)
        if "CTH" in index[0]:
            analyte_type.append(2)
    return (
        np.asarray(organized_data),
        np.asarray(analyte_type),
        np.asarray(concentration),
        np.asarray(stratify_groups),
    )


def num_unique_groups(stratify_groups):
    return len(np.unique(stratify_groups))


def get_train_test_group_splits(stratify_groups, test_size=0.2):
    """Get the train and test indices for the dataset while
    taking into account the stratify groups"""
    indices = np.arange(len(stratify_groups))
    train_indices, test_indices = next(
        GroupShuffleSplit(test_size=test_size, n_splits=1).split(
            indices, groups=stratify_groups
        )
    )
    return train_indices, test_indices


class EGFETDataset(Dataset):
    def __init__(
        self,
        path: Path,
        exclude_experiment: List[str] = [],
        exclude_concentration: List[str] = [],
        include_only: List[str] = [],
        downsample: Optional[int] = None,
        verbose_loading: int = 0,
    ):
        df = load_egfet_dataset(
            path,
            exclude_experiment=exclude_experiment,
            exclude_concentration=exclude_concentration,
            load_only=include_only,
            downsample=downsample,
            verbose_loading=verbose_loading,
        )
        self.data, self.target, self.concentration, self.stratify_groups = prepare_data(
            df
        )
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.target = torch.tensor(self.target, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

    @classmethod
    def build_dataset_splits(
        cls,
        path: Path,
        exclude_experiment: List[str] = [],
        exclude_concentration: List[str] = [],
        include_only: List[str] = [],
        downsample: Optional[int] = None,
        verbose_loading: int = 0,
        test_size: float = 0.2,
    ):
        """Build the train and test datasets from the given path

        Args:
            path (Path): The path to the data
            exclude_experiment (List[str]): The experiments to exclude
            exclude_concentration (List[str]): The concentrations to exclude
            include_only (List[str]): The experiments to include
            downsample (Optional[int]): The number of samples to downsample to
            verbose_loading (int): The verbosity of the loading process
            test_size (float): The size of the test dataset


        Returns:
            Tuple[Dataset, Dataset]: The train and test datasets
        """
        dataset = cls(
            path,
            exclude_experiment=exclude_experiment,
            exclude_concentration=exclude_concentration,
            include_only=include_only,
            downsample=downsample,
            verbose_loading=verbose_loading,
        )
        train_indices, test_indices = get_train_test_group_splits(
            dataset.stratify_groups, test_size=test_size
        )
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        return train_dataset, test_dataset

    @classmethod
    def yield_dataset_folds(
        cls,
        path: Path,
        exclude_experiment: List[str] = [],
        exclude_concentration: List[str] = [],
        include_only: List[str] = [],
        downsample: Optional[int] = None,
        verbose_loading: int = 0,
        n_splits: int = 5,
        n_repeats: int = 1,
    ):
        """Yield the train and test datasets for the given path

        Args:
            path (Path): The path to the data
            exclude_experiment (List[str]): The experiments to exclude
            exclude_concentration (List[str]): The concentrations to exclude
            include_only (List[str]): The experiments to include
            downsample (Optional[int]): The number of samples to downsample to
            verbose_loading (int): The verbosity of the loading process
            n_splits (int): The number of splits
            n_repeats (int): The number of repeats

        Yields:
            Tuple[Dataset, Dataset]: The train and test datasets
        """
        dataset = cls(
            path,
            exclude_experiment=exclude_experiment,
            exclude_concentration=exclude_concentration,
            include_only=include_only,
            downsample=downsample,
            verbose_loading=verbose_loading,
        )
        splitter = RepeatedGroupKFold(n_splits=n_splits, n_repeats=n_repeats)
        for train_indices, test_indices in splitter.split(
            dataset.data, groups=dataset.stratify_groups
        ):
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            test_dataset = torch.utils.data.Subset(dataset, test_indices)
            yield train_dataset, test_dataset

class EGFETDatasetBuilder(AbstractDatasetBuilder):
    name = "egfet"

    def build(
        self,
        train_transforms:MultiArgCompose = None,
        val_transforms:MultiArgCompose = None,
        **config,
    ) -> EGFETDataset:
        dataset_name = config["name"]
        dataset_params = config["args"]
        dataset = self._builders[dataset_name](**dataset_params)
        return dataset
    