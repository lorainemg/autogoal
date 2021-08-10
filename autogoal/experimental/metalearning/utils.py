import numpy as np
import sys
from functools import reduce
from operator import mul

from autogoal.experimental.metalearning.datasets import DatasetExtractor, Dataset
from sklearn.model_selection import train_test_split
from typing import Tuple, List
from pathlib import Path


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