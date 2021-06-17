from pathlib import Path
from importlib import import_module
from typing import List
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.io import arff
from itertools import chain
import pandas as pd
import numpy as np


from autogoal.kb import (
    SemanticType,
    Supervised,
    Seq,
    Word,
    Label,
    VectorCategorical,
    VectorDiscrete,
    Document,
    Sentence
)

# Class that represent a dataset.


class Dataset:
    def __init__(self, name: str, load, path=None):
        self.name = name
        self.input_type = None
        self.output_type = None
        self._load = load if path is None else load(path)

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
        self._check_input()
        self._add_supervised_as_input()
        return self.input_type

    def _infer_output_type(self, y):
        self.output_type = SemanticType.infer(y)
        self._check_output()

    def _check_input(self):
        """
        Check input to be valid according to the avalaible algorithms
        """
        # For now, there is no avalaible algorithm for sequence of documents
        # therefore, this type will be changed
        if issubclass(Seq[Document], self.input_type):
            self.input_type = Seq[Sentence]

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
        """
        if dataset_folder is None:
            dataset_folder = Path('autogoal/datasets')
        self.datasets: List[Dataset] = self.get_datasets(dataset_folder)

    @staticmethod
    def extract_py_dataset(path_obj) -> Dataset:
        """
        Expected format: python scripts with a `load()` function
        """
        name = path_obj.name[:-3]
        if name in ('movie_reviews', 'gisette'):
            return None
        try:
            mod = import_module(f'.{name}', '.'.join(path_obj.parts[:-1]))
            return Dataset(name, mod.load)
        except:
            return None

    @staticmethod
    def extract_arff_dataset(path_obj) -> Dataset:
        name = path_obj.name[:-5]
        return Dataset(name, load(path_obj))

    def get_datasets(self, dataset_folder: Path):
        datasets = []
        for fn in dataset_folder.glob('**/*'):
            if fn.is_file():
                d = None
                if fn.suffix == '.py':
                    d = self.extract_py_dataset(fn)
                elif fn.suffix == '.arff':
                    d = self.extract_arff_dataset(fn)
                if d is not None:
                    datasets.append(d)
            elif fn.is_file():
                datasets += self.get_datasets(fn)
        return datasets


def load(path):
    """Loader method of arff files"""
    def wrapper(*args, **kwargs):
        data, metadata = arff.loadarff(path)
        df = pd.DataFrame(data)

        # Get the target name
        for attr_name in metadata.names():
            if attr_name.lower() in ('class', 'target', 'y', 'binaryClass'):
                target = attr_name
                break
        else:
            target = metadata.names()[-1]

        # Put the type of y correctly
        dtype = 'U' if metadata[target][0] == 'nominal' else 'f'
        y = np.array(df[target], dtype=dtype)

        X = df.loc[:, df.columns != target]
        # Transform nominal values into integer ones
        for column in list(X):
            if metadata[column][0] == 'nominal':
                X.loc[:, column] = LabelEncoder().fit_transform(X[column])
        X = np.array(X, dtype='f')

        return X, y
    return wrapper


