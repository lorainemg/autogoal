import re
from pathlib import Path
from random import shuffle
from typing import List
from zipfile import ZipFile, ZIP_DEFLATED

from autogoal.experimental.metalearning import DatasetExtractor, Dataset
from autogoal.experimental.metalearning.metalearner import MetaLearner
from autogoal.experimental.metalearning.xgb_learner import XGBRankerMetaLearner
from autogoal.experimental.metalearning.nn_metalearner import NNMetaLearner
from autogoal.experimental.metalearning.metafeatures import MetaFeatureExtractor
from autogoal.experimental.metalearning.results_logger import ResultsLogger
from autogoal.experimental.metalearning.utils import MTL_RESOURCES_PATH
from autogoal.experimental.metalearning.distance_measures import l1_distance, l2_distance


from autogoal.kb import Supervised, Tensor, Continuous, Dense, Categorical
from autogoal.ml import AutoML
from autogoal.utils import Min

# from download_datasets import download_classification_datasets
import os
import json
import numpy as np
# from autogoal.experimental.metalearning.experiments import datasets_feat

err_file_path: Path = Path(MTL_RESOURCES_PATH) / 'errors.txt'

err_text = []


def test_automl(datasets: List[Dataset], iterations: int = 1, visited_datasets=None):
    """Tests automl using autogoal"""
    visited_datasets = visited_datasets or []
    for i in range(iterations):
        for dataset in datasets:
            if dataset.name in visited_datasets:
                continue

            X, y = dataset.load()
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

            automl = AutoML(
                    input=(Tensor[2, Continuous, Dense],
                           Supervised[Tensor[1, Categorical, Dense]]),
                    output=Tensor[1, Categorical, Dense],
                    evaluation_timeout=5 * Min,
                    search_timeout=10 * Min)
            name = f'automl_{dataset.name}_{i}'
            try:
                automl.fit(X, y, logger=ResultsLogger('autogoal', name))
            except Exception as e:
                with err_file_path.open('a') as fd:
                    fd.write(f'Error in dataset {dataset.name} in test_automl method \n\t{e}\n')


def split_datasets(datasets: List[Dataset], train_proportion: float, random=True):
    """Splits datasets """
    if random:
        shuffle(datasets)
    train_size = int(len(datasets) * train_proportion)
    train_set = datasets[:train_size]
    test_set = datasets[train_size:]
    return train_set, test_set


def inspect_datasets(datasets: List[Dataset]):
    """Tests dataset object: load and input/output type"""
    review_datasets = []
    for d in datasets:
        print(d.name)
        X, y = d.load()
        if X is None or y is None:
            continue
        print(f'input_type: {d.input_type}')
        print(f'output_type: {d.output_type}')
        review_datasets.append(d)
    return review_datasets


def test_mtl(train_dataset: List[Dataset], test_dataset: List[Dataset], learner: MetaLearner, iterations: int = 1):
    """Tests metalearning methods"""
    learner.meta_train(train_dataset)
    for _ in range(iterations):
        # learner.test(test_dataset)
        try:
            learner.evaluate_datasets(test_dataset)
        except Exception as e:
            with err_file_path.open('a') as fd:
                fd.write(f'Error in test_mtl method \n \t{e}\n')


def test_autogoal_with_mtl(datasets: List[Dataset], learner: MetaLearner, iterations: int = 1):
    for i in range(iterations):
        for dataset in datasets:
            X, y = dataset.load()
            automl = AutoML(
                input=dataset.input_type,
                output=dataset.output_type,
                evaluation_timeout=5 * Min,
                search_timeout=10 * Min,
                metalearner=learner)
            name = f'mtl_{dataset.name}_{i}'
            try:
                automl.fit(X, y, name=dataset.name, logger=ResultsLogger(learner.name, name))
            except Exception as e:
                with err_file_path.open('a') as fd:
                    fd.write(f'Error in {dataset.name} in test_autogoal_with_mtl method \n\t{e}\n\n')


def compress_resources(zip_path: str = 'resources.zip'):
    """Compress resources in the resources path"""
    root_path = Path(MTL_RESOURCES_PATH)

    with ZipFile(zip_path, 'w') as zip_obj:
        for folder_name, subfolders, filenames in os.walk(root_path):
            for filename in filenames:
                # Create complete filepath of file in directory
                file_path = os.path.join(folder_name, filename)
                # Add file to zip
                zip_obj.write(file_path, file_path, compress_type=ZIP_DEFLATED)


def leave_one_out(datasets, learners):
    """Test in a leave one out manner"""
    shuffle(datasets)

    for i, ds in enumerate(datasets):
        train_datasets = []
        if i + 1 < len(datasets):
            train_datasets.extend(datasets[i + 1:])
        if i - 1 > 0:
            train_datasets.extend(datasets[: i - 1])

        test_automl([ds], 1)

        for learner in learners:
            learner.meta_train(train_datasets)
            test_mtl(train_datasets, [ds], learner, 1)
            test_autogoal_with_mtl([ds], learner, 1)


def get_evaluated_datasets():
    autogoal_path = Path(MTL_RESOURCES_PATH) / 'results' / 'autogoal'

    visited_datasets = []
    if autogoal_path.exists():
        results = get_datasets_in_path(autogoal_path)
        dataset_re = re.compile('\w+_(\d+)_\d+')
        for ds in results:
            m = dataset_re.match(ds)
            if m is not None:
                dataset = m.group(1)
                visited_datasets.append(dataset)
    return visited_datasets


def cv(datasets, learners):
    train_dataset, test_dataset = split_datasets(datasets, 0.75)

    # visited_datasets = get_evaluated_datasets()

    test_automl(test_dataset, 1)
    for learner in learners:
        learner.meta_train(train_dataset)
        # test_mtl(train_dataset, test_dataset, learner, 1)
        test_autogoal_with_mtl(test_dataset, learner, 3)


def save_metafeatures(datasets: List[Dataset]):
    mfe = MetaFeatureExtractor()
    p = Path(MTL_RESOURCES_PATH) / "mfeat"
    p.mkdir(exist_ok=True, parents=True)
    visited_datasets = get_datasets_in_path(p)
    for ds in datasets:
        try:
            if ds.name in visited_datasets:
                continue
            X, y = ds.load()
            # metafeatures = number_of_classes(X, y)
            metafeatures = mfe.extract_features(X, y, ds)
            json.dump({
                'metafeatures': metafeatures
            }, open(f'{p  / ds.name}.json', 'w'))
        except Exception as e:
            print(e)
            with err_file_path.open('a') as fd:
                fd.write(f'Error in {ds.name} in save_metafeatures method \n\t{e}\n\n')


def get_datasets_in_path(path: Path):
    """
    Gets all the dataset with stored information in a specific path.
    This is used to check which datasets has features extracted.
    """
    return set(file.name[:-5] for file in path.glob('*.json'))


if __name__ == '__main__':
    if not err_file_path.exists():
        Path(MTL_RESOURCES_PATH).mkdir(parents=True, exist_ok=True)
        err_file_path.open('x').close()

    datasets = DatasetExtractor(Path('datasets')).datasets
    print(len(datasets))

    datasets = inspect_datasets(datasets)
    print(len(datasets))

    # download_classification_datasets()
    # save_metafeatures(datasets)

    datasets, _ = split_datasets(datasets, 0.8)
    print(len(datasets))

    xgb_ranker_l2 = XGBRankerMetaLearner(distance_metric=l2_distance, load=False, learner_name='xgb_l2')
    nn_learner = NNMetaLearner(distance_metric=l2_distance, strategy='simple', learner_name='nn_simple_l2')
    nn_learner_aggregated = NNMetaLearner(distance_metric=l2_distance, learner_name='nn_aggregated_l2')

    # All datasets are trained to get the meta-features of the problem
    # nn_learner.train(datasets)
    cv(datasets, [xgb_ranker_l2, nn_learner, nn_learner_aggregated])

    with err_file_path.open('a') as fd:
        fd.write(f'----------------------------------------------------')
