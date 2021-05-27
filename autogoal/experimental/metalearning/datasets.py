from pathlib import Path
from importlib import import_module
from typing import List
from sklearn.model_selection import train_test_split
import numpy as np


class Dataset:
    def __init__(self, name: str, load):
        self.name = name
        self._load = load
        self.X, self.y = None, None
        self.load()

    def load(self, *args, **kwargs):
        result = self._load(*args, **kwargs)
        if len(result) == 1:
            self.X = result
            self.y = None
        elif len(result) == 2:
            self.X, self.y = result
        elif len(result) == 4:
            X_train, y_train, X_test, y_test = result
            self.X = np.concatenate((X_train, X_test))
            self.y = np.concatenate((y_train, y_test))
        return self.X, self.y


class DatasetExtractor:
    def __init__(self, dataset_folder=None):
        """
        Extracts the datasets from a given folder
        Expected format: python scripts with a `load()` function
        """
        if dataset_folder is None:
            dataset_folder = Path('autogoal/datasets')
        self.datasets: List[Dataset] = self.get_datasets(dataset_folder)

    def get_datasets(self, dataset_folder: Path) -> List[Dataset]:
        datasets = []
        for fn in dataset_folder.glob('*.py'):
            name = fn.name[:-3]
            try:
                mod = import_module(f'.{name}', '.'.join(fn.parts[:-1]))
                datasets.append(Dataset(name, mod.load))
            except:
                pass
        return datasets