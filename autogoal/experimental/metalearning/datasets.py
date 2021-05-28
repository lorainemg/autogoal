from pathlib import Path
from importlib import import_module
from typing import List
from sklearn.model_selection import train_test_split
import numpy as np


class Dataset:
    def __init__(self, name: str, load):
        self.name = name
        self._load = load

    def load(self, *args, **kwargs):
        result = self._load(*args, **kwargs)
        X_train, y_train, X_test, y_test = None, None, None, None
        if len(result) == 1:
            X_train, X_test = train_test_split(result, test_size=0.15)
            y_train, y_test = None, None
        elif len(result) == 2:
            X, y = result
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
        elif len(result) == 4:
            X_train, y_train, X_test, y_test = result
            # try:
            #     X = np.concatenate((X_train, X_test))
            # except ValueError:
            #     X = np.concatenate((X_train.todense(), X_test.todense()))
            # try:
            #     y = np.concatenate((y_train, y_test))
            # except ValueError:
            #     y = np.concatenate((y_train.todense(), y_test.todense()))
        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


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