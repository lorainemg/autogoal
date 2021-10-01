from pathlib import Path
from random import shuffle
from typing import List
from zipfile import ZipFile, ZIP_DEFLATED

from autogoal.experimental.metalearning import DatasetExtractor, Dataset
from autogoal.experimental.metalearning.metalearner import MetaLearner
from autogoal.experimental.metalearning.xgb_learner import XGBRankerMetaLearner
from autogoal.experimental.metalearning.nn_metalearner import NNMetaLearner
from autogoal.experimental.metalearning.results_logger import ResultsLogger
from autogoal.experimental.metalearning.utils import MTL_RESOURCES_PATH

from autogoal.kb import Supervised, Tensor, Continuous, Dense, Categorical
from autogoal.ml import AutoML
from autogoal.utils import Min

from download_datasets import download_classification_datasets
import os
# from autogoal.experimental.metalearning.experiments import datasets_feat

err_file_path: Path = Path(MTL_RESOURCES_PATH) / 'errors.txt'

err_text = []

def test_automl(datasets: List[Dataset], iterations: int = 1):
    """Tests automl using autogoal"""
    for i in range(iterations):
        for dataset in datasets:
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
                    fd.write(f'Error in dataset {dataset.name} in test_automl method \n \t{e}\n')

    # print(automl.best_pipeline_)
    # print(automl.best_score_)
    #
    # score = automl.score(X_test, y_test)
    # print(f"Score: {score:0.3f}")
    #
    # predictions = automl.predict(X_test)
    #
    # for sentence, real, predicted in zip(X_test[:10], y_test, predictions):
    #     print(sentence, "-->", real, "vs", predicted)


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
            # test_mtl(train_datasets, [ds], learner, 1)
            test_autogoal_with_mtl([ds], learner, 1)


def cv(datasets, learners):
    train_dataset, test_dataset = split_datasets(datasets, 0.75)
    # train_dataset, test_dataset = datasets[:60], datasets[60:]

    test_automl(test_dataset, 1)
    for learner in learners:
        learner.meta_train(train_dataset)
        # test_mtl(train_dataset, test_dataset, learner, 1)
        test_autogoal_with_mtl(test_dataset, learner, 1)


if __name__ == '__main__':
    if not err_file_path.exists():
        Path(MTL_RESOURCES_PATH).mkdir(parents=True, exist_ok=True)
        err_file_path.open('x').close()

    # datasets = DatasetExtractor(Path('/home/coder/.autogoal/data/classification/lt 5000')).datasets

    # download_classification_datasets()
    datasets = DatasetExtractor(Path('datasets/classification')).datasets
    print(len(datasets))

    datasets = inspect_datasets(datasets)

    xgb_ranker = XGBRankerMetaLearner()
    nn_learner = NNMetaLearner()
    # All datasets are trained to get the meta-features of the problem
    xgb_ranker.train(datasets)

    datasets, _ = split_datasets(datasets, 0.8)

    # leave_one_out(datasets, [xgb_ranker, nn_learner])
    cv(datasets, [xgb_ranker, nn_learner])

    with err_file_path.open('a') as fd:
        fd.write(f'----------------------------------------------------')


