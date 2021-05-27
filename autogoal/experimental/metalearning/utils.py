from autogoal.experimental.metalearning.datasets import DatasetExtractor, Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, List


def get_train_test_datasets(dataset_folder: Path = None) -> Tuple[List[Dataset], List[Dataset]]:
    dataset_ext = DatasetExtractor(dataset_folder)
    return train_test_split(dataset_ext)
