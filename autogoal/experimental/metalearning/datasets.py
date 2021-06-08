from pathlib import Path
from importlib import import_module
from typing import List
from sklearn.model_selection import train_test_split

from autogoal.kb import (
    SemanticType,
    Supervised,
    Seq,
    Word,
    Label,
    VectorCategorical,
    VectorDiscrete
)

# Class that represent a dataset.


class Dataset:
    def __init__(self, name: str, load):
        self.name = name
        self._load = load
        self.input_type = None
        self.output_type = None

    def load(self, *args, **kwargs):
        """Loads the dataset, returning the train and test collection"""
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

        if self.input_type is None:
            self.infer_types(X_train, y_train)

        return X_train, y_train, X_test, y_test

    def infer_types(self, X, y):
        """Infers the semantic type of the input and the output"""
        self._infer_output_type(y)
        # input type depends on output type, then its processed second
        self._infer_input_type(X)

    def _infer_input_type(self, X):
        self.input_type = SemanticType.infer(X)
        self._add_supervised_as_input()
        return self.input_type

    def _infer_output_type(self, y):
        self.output_type = SemanticType.infer(y)
        self._check_output()

    def _add_supervised_as_input(self):
        """
        Add Supervised annotation to input to supervised algorithms.

        This is not done with Semantic infer and is necessary to build pipelines.
        """
        if self.output_type and not issubclass(Supervised, self.input_type):
            try:
                if all([not issubclass(in_type, Supervised) for in_type in self.input_type]):
                    self.input_type = tuple(list(self.input_type) + [Supervised[self.output_type]])
            except TypeError:
                self.input_type = (self.input_type, Supervised[self.output_type])

    def _check_output(self):
        """
        Semantic infer returns Seq[Word] to a sequence of labels, this is fixed.

        Semantic infer never matches with Seq[Label] and this is necessary to allow the algorithms to work properly.
        """
        if issubclass(Seq[Seq[Word]], self.output_type):
            self.output_type = Seq[Seq[Label]]
        elif issubclass(Seq[Word], self.output_type):
            self.output_type = Seq[Label]
        elif issubclass(VectorDiscrete, self.output_type):
            self.output_type = VectorCategorical


# Dataset extractor that extracts all .py files in a given folder
# with a method `load`


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
            if name in ('haha', 'movie_reviews', 'gisette'):
                continue
            try:
                mod = import_module(f'.{name}', '.'.join(fn.parts[:-1]))
                datasets.append(Dataset(name, mod.load))
            except:
                pass
        return datasets
