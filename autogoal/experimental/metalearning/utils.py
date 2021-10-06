import numpy as np
import pandas as pd
import sys
from functools import reduce
from operator import mul
from autogoal.experimental.metalearning.datasets import DatasetExtractor, Dataset
from sklearn.model_selection import train_test_split
from typing import Tuple, List
from pathlib import Path

MTL_RESOURCES_PATH = 'autogoal/experimental/metalearning/resources/'

def get_train_test_datasets(dataset_folder: Path = None) -> Tuple[List[Dataset], List[Dataset]]:
    dataset_ext = DatasetExtractor(dataset_folder)
    return train_test_split(dataset_ext)


def reduce_shape(X):
    """If X has more than 2-D, then the X is reshape to convert it in 2-D array"""
    dimension = reduce(mul, X.shape[1:], 1)
    return np.reshape(X, (X.shape[0], dimension))


def pad_arrays(array, max_len):
    """Padds a array to have length=max_len"""
    length = max_len - len(array)
    return np.pad(array, (0, length), mode='empty')


def fix_indef_values(vect):
    """Substitutes nan and inf values"""
    vect[np.isnan(vect)] = 0
    vect[np.isinf(vect)] = sys.float_info.max


def get_numerical_features(df: pd.DataFrame):
    numerical_types = 'iuf'     # signed and unsigned ints, floats, complex number
    g = df.columns.to_series().groupby(df.dtypes).groups
    columns = []
    for k, v in g.items():
        if k.kind in numerical_types:
            columns.extend(v)
    return columns


def train_test_split(X, y):
    len_x = (len(X)
        if isinstance(X, list) or isinstance(X, tuple)
        else X.shape[0])
    indices = np.arange(0, len_x)
    np.random.shuffle(indices)
    split_index = int(0.7 * len(indices))
    train_indices = indices[:-split_index]
    test_indices = indices[-split_index:]
    if isinstance(X, list) or isinstance(X, tuple):
        X_train, y_train, X_test, y_test = (
            [X[i] for i in train_indices],
            y[train_indices],
            [X[i] for i in test_indices],
            y[test_indices],
        )
    else:
        X_train, y_train, X_test, y_test = (
            X[train_indices],
            y[train_indices],
            X[test_indices],
            y[test_indices],
        )
    return X_train, y_train, X_test, y_test
