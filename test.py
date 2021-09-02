from autogoal.ml import AutoML
from autogoal.kb import MatrixContinuousDense, VectorCategorical, Supervised, Tensor, Continuous, Dense, Categorical
from typing import List

from sklearn.model_selection import train_test_split
from autogoal.search import RichLogger
from autogoal.experimental.metalearning.xgb_ranker import XGBRankerMetaLearner
from autogoal.experimental.metalearning import DatasetExtractor, Dataset
from autogoal.experimental.metalearning.results_logger import ResultsLogger
from autogoal.datasets import cars
from pathlib import Path
from random import shuffle
from autogoal.utils import Min


def test_automl(X, y):
    """Tests automl using autogoal"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    automl = AutoML(
            input=(Tensor[2, Continuous, Dense],
                   Supervised[Tensor[1, Categorical, Dense]]),
            output=Tensor[1, Categorical, Dense],
            evaluation_timeout=1 * Min,
            search_timeout=10 * Min,
            metalearner=XGBRankerMetaLearner()
    )

    automl.fit(X_train, y_train, logger=ResultsLogger('cars'), name='cars')

    print(automl.best_pipeline_)
    print(automl.best_score_)

    score = automl.score(X_test, y_test)
    print(f"Score: {score:0.3f}")

    predictions = automl.predict(X_test)

    for sentence, real, predicted in zip(X_test[:10], y_test, predictions):
        print(sentence, "-->", real, "vs", predicted)


def split_datasets(datasets: List[Dataset], proportion: float):
    """Splits datasets """
    shuffle(datasets)
    train_size = int(len(datasets) * proportion)
    train_set = datasets[train_size:]
    test_set = datasets[:train_size]
    return train_set, test_set


def test_datasets(datasets: List[Dataset]):
    """Tests dataset object: load and input/output type"""
    for d in datasets:
        print(d.name)
        X, y = d.load()
        print(f'input_type: {d.input_type}')
        print(f'output_type: {d.output_type}')


def test_mtl(train_dataset: List[Dataset], test_dataset: List[Dataset], iterations: int = 1):
    """Tests metalearning methods"""
    for _ in range(iterations):
        learner = XGBRankerMetaLearner()
        learner.meta_train(train_dataset)
        learner.test(test_dataset)
        learner.evaluate_datasets(test_dataset)


def test_autogoal_with_mtl(datasets: List[Dataset], iterations: int = 1):
    for i in range(iterations):
        for dataset in datasets:
            X, y = dataset.load()
            automl = AutoML(
                input=dataset.input_type,
                output=dataset.output_type,
                evaluation_timeout=1 * Min,
                search_timeout=10 * Min,
                metalearner=XGBRankerMetaLearner())
            automl.fit(X, y, name=dataset.name, logger=ResultsLogger(dataset.name + str(i)))


if __name__ == '__main__':
    # X, y = cars.load()
    # test_automl(X, y)

    datasets = DatasetExtractor(Path('/home/coder/.autogoal/data/classification/lt 5000')).datasets
    # test_datasets(datasets)
    train_dataset, test_dataset = split_datasets(datasets, 0.15)
    test_mtl(train_dataset, test_dataset[:1], 1)
    test_autogoal_with_mtl(test_dataset[:1], 1)
