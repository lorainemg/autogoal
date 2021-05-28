from autogoal.experimental.metalearning.metafeatures import MetaFeatureExtractor
from autogoal.experimental.metalearning.datasets import Dataset
from autogoal.contrib import find_classes
from autogoal.ml import AutoML

from typing import List


class MetaLearner:
    def __init__(self, k=5, features_extractor=None):
        self.k = k  # the numbers of possible algorithms to predict
        self.metafeature_extractor = MetaFeatureExtractor(features_extractor)

    def train(self, datasets: List[Dataset]):
        raise NotImplementedError

    def predict(self, dataset: Dataset):
        raise NotImplementedError

    def test(self, datasets: List[Dataset]):
        return [self.predict(dataset) for dataset in datasets]

    def extract_metafeatures(self, datasets: List[Dataset]):
        """Extracts the features of the datasets"""
        return [self._extract_metafeatures(d) for d in datasets]

    def _extract_metafeatures(self, dataset: Dataset):
        X, y, _, _ = dataset.load()
        return self.metafeature_extractor.extract_features(X, y)

    def extract_metatargets(self, datasets: List[Dataset], algorithms=None):
        """
        Extracts the features of the solution.
        For this, a list of algorithms is trained with the datasets.
        """
        return [self._extract_metatargets(d, algorithms) for d in datasets]

    def _extract_metatargets(self, dataset: Dataset, algorithms):
        """
        Hay 3 formas posibles de entrenar:
        1. Entrenar con todos los algoritmos con sus parámetros por defecto
        2. Entrenanr con todoos los posibles parámetros a maximizar
        3. Entrenar usando automl para buscar el mejor pipeline.
        Esta versión usará la 3ra opción.
        """
        X_train, y_train, X_test, y_test = dataset.load()
        automl = AutoML(registry=algorithms)
        try:
            automl.fit(X_train, y_train)
            # creo que lo ideal no sería coger el mejor pipeline, si no los k mejores.
            score = automl.score(X_test, y_test)
            # todo: buscar una manera de representar los pipeline
            return {
                'pipeline': automl.best_pipeline_,
                'score': score
            }
        except TypeError as e:
            print(f'Error {dataset.name}: {e}')

