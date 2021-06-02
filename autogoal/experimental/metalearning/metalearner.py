from autogoal.experimental.metalearning.metafeatures import MetaFeatureExtractor
from autogoal.experimental.metalearning.datasets import Dataset
from autogoal.ml import AutoML

from sklearn.feature_extraction import DictVectorizer
from pathlib import Path
from typing import List
import pickle
import json


class MetaLearner:
    def __init__(self, k=5, features_extractor=None):
        self.k = k  # the numbers of possible algorithms to predict
        self.metafeature_extractor = MetaFeatureExtractor(features_extractor)
        self.vectorizer = DictVectorizer()
        self._vectorizer_path = Path('metafeatures.vct')
        self._features_vect_path = Path('metafeatures_vect.pkl')

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

        Returns a list of the labels with the best pipelines and
        a list of metatargets (features of the training process)
        """
        meta_labels = []
        meta_targets = []
        for d in datasets:
            meta_label, meta_target = self._extract_metatargets(d, algorithms)
            meta_labels.append(meta_label)
            meta_targets.append(meta_target)
        return meta_labels, meta_targets

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
            # missing more metatargets, maybe training time
            # more features about the pipeline...
            return automl.best_pipeline_, {
                'score': score
            }
        except TypeError as e:
            print(f'Error {dataset.name}: {e}')

    def preprocess_features(self, metafeatures: List[dict], metatargets: List[dict], training=False):
        features = []
        for meta_feat, meta_tar in zip(metafeatures, metatargets):
            meta_feat.update(meta_tar)
            features.append(meta_feat)
        if training:
            self.vectorizer.fit(features)
        return self.vectorizer.transform(features)

    # def get_training_samples(self, datasets: List[Dataset]):
    #     """
    #     Returns all the features vectorized and the labels.
    #     """
    #     metafeatures = self.extract_metafeatures(datasets)
    #     json.dump(metafeatures, open(Path('metafeatures.json'), 'w'))
    #     metalabels, metatargets = self.extract_metatargets(datasets)
    #     return self.preprocess_features(metafeatures, metatargets, training=True), metalabels

    def get_training_samples(self, datasets: List[Dataset]):
        """
        Returns all the features vectorized and the labels.
        """
        if not self._features_vect_path.exists():
            metafeatures = self.extract_metafeatures(datasets)
            metafeatures = self.preprocess_metafeatures(metafeatures)
            self.save_training_metafeatures(metafeatures)
        else:
            metafeatures = self.load_training_metafeatures()
        metalabels, _ = self.extract_metatargets(datasets)
        return metafeatures, metalabels

    def preprocess_metafeatures(self, metafeatures):
        if not self._vectorizer_path.exists():
            self.vectorizer.fit(metafeatures)
            self.save_vectorizer()
        else:
            self.load_vectorizer()
        return self.vectorizer.transform(metafeatures)

    def preprocess_metafeature(self, dataset: Dataset):
        metafeature = self._extract_metafeatures(dataset)
        return self.vectorizer.transform([metafeature])[0]

    def save_vectorizer(self):
        pickle.dump(self.vectorizer, open(self._vectorizer_path, 'wb'))

    def load_vectorizer(self):
        self.vectorizer = pickle.load(open(self._vectorizer_path, 'rb'))

    def save_training_metafeatures(self, metafeatures):
        pickle.dump(metafeatures, open(self._features_vect_path, 'wb'))

    def load_training_metafeatures(self):
        return pickle.load(open(self._features_vect_path, 'rb'))


