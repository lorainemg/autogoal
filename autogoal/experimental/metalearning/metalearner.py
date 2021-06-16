from autogoal.experimental.metalearning.metafeatures import MetaFeatureExtractor
from autogoal.experimental.metalearning.datasets import Dataset
from autogoal.experimental.metalearning.utils import pad_arrays
from autogoal.ml import AutoML
from autogoal.search import RichLogger
from autogoal.utils import Hour, Min

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from itertools import chain
from pathlib import Path
from typing import List
import numpy as np
import json


class MetaLearner:
    def __init__(self, k=5, features_extractor=None):
        self.k = k  # the numbers of possible algorithms to predict
        self.meta_feature_extractor = MetaFeatureExtractor(features_extractor)
        self._vectorizer = DictVectorizer()
        self._features_path = Path('autogoal/experimental/metalearning/resources/meta_features.json')
        self._pipelines_path = Path('autogoal/experimental/metalearning/resources/meta_labels.json')
        self._scores_path = Path('autogoal/experimental/metalearning/resources/meta_targets.json')
        self._pipelines_encoder = LabelEncoder()

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
        return self.meta_feature_extractor.extract_features(X, y)

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
        2. Entrenar con todoos los posibles parámetros a maximizar
        3. Entrenar usando automl para buscar el mejor pipeline.
        Esta versión usará la 3ra opción.
        """
        X_train, y_train, X_test, y_test = dataset.load()
        automl = AutoML(
            input=dataset.input_type,
            output=dataset.output_type,
            registry=algorithms,
            evaluation_timeout=5 * Min,
            search_timeout=20 * Min,
            search_iterations=1000
        )
        try:
            automl.fit(X_train, y_train, logger=RichLogger())
            # creo que lo ideal no sería coger el mejor pipeline, si no los k mejores.
            score = automl.score(X_test, y_test)
            # missing more metatargets, maybe training time

            features, features_types = self.extract_pipelines_features(automl.best_pipeline_)
            dict_ = {'features': features, 'features_types': features_types, 'name': dataset.name}
            return dict_, score
        except TypeError as e:
            print(f'Error {dataset.name}: {e}')
            return None, None

    def extract_pipelines_features(self, solution):
        """Extracts the features of the pipelines in a comprehensive manner"""
        sampler = solution.sampler_
        features = {k: v for k, v in sampler._updates.items() if isinstance(k, str)}
        feature_types = {k: repr(v) for k, v in sampler._model.items() if k in features}
        return features, feature_types

    def get_training_samples(self, datasets: List[Dataset]):
        """
        Returns all the features vectorized and the labels.
        """
        # Extracts meta_features, a processing of the dataset is done
        if not self._features_path.exists():
            meta_features = self.extract_metafeatures(datasets)
            self.save_training_metafeatures(meta_features)
        else:
            meta_features = self.load_training_metafeatures()

        # Extracts meta_labels (pipelines) and meta_targets (scores)
        # For this, training is done across all datasets.
        if not self._pipelines_path.exists():
            meta_labels, meta_targets = self.extract_metatargets(datasets)
            self.save_training_metalabels(meta_labels, meta_targets)
        else:
            meta_labels, meta_targets = self.load_training_metalabels()

        # Preprocess meta_labels and meta_features to obtain a vector-like meta_features
        meta_features = self.preprocess_datasets(meta_features)
        meta_labels = self.preprocess_pipelines(meta_labels)
        return meta_features, meta_labels, meta_targets

    def append_features_and_labels(self, meta_features, meta_labels):
        """
        Appends the matrix of meta_features and meta_labels to create a join matrix
        where the labels columns (corresponding to the pipelined algorithms)
        have to be filled for a new datasets.
        """
        try:
            features = meta_features.tolist()
        except AttributeError:
            features = list(meta_features)
        for i in range(len(features)):
            features[i].extend(meta_labels[i])
        return np.array(features)

    def create_duplicate_data_features(self, data_features, n):
        """
        Repeats n times the data_features to obtain various instances of the same list.

        Every instance of this features will be joined with a different pipeline.
        """
        data_features.tolist()
        return [list(data_features) for _ in range(n)]

    def preprocess_datasets(self, meta_features):
        self._vectorizer.fit(meta_features)
        vect = np.array(self._vectorizer.transform(meta_features).todense())
        vect[np.isnan(vect)] = 0
        return vect

    def preprocess_pipelines(self, meta_labels):
        pipelines = []
        max_len = 0
        for meta_label in meta_labels:
            features = meta_label['features']
            pipeline = []
            for algorithm, param in features.items():
                if algorithm == 'End':
                    break
                # First position in param is the amount of times the algorithm is applied (I think)
                for i in range(param[0]):
                    pipeline.append(algorithm)
            max_len = max(max_len, len(pipeline))
            pipelines.append(pipeline)
        # padds the pipelines so every pipeline has the same length
        padded_pipelines = np.array([pad_arrays(pipeline, max_len) for pipeline in pipelines])
        # Encodes the pipelines names
        self._pipelines_encoder.fit(list(chain.from_iterable(padded_pipelines)))
        return [self._pipelines_encoder.transform(p) for p in padded_pipelines]

    def preprocess_metafeature(self, dataset: Dataset):
        meta_feature = self._extract_metafeatures(dataset)
        vect = np.array(self._vectorizer.transform([meta_feature]).todense())[0]
        vect[np.isnan(vect)] = 0
        return vect

    def decode_pipelines(self, pipelines: List[List[int]]) -> List[List[str]]:
        return [self._pipelines_encoder.inverse_transform(p) for p in pipelines]

    def save_training_metafeatures(self, meta_features):
        json.dump(meta_features, open(self._features_path, 'w'))

    def load_training_metafeatures(self):
        return json.load(open(self._features_path, 'r'))

    def save_training_metalabels(self, meta_labels, meta_targets):
        json.dump(meta_labels, open(self._pipelines_path, 'w'))
        json.dump(meta_targets, open(self._scores_path, 'w'))

    def load_training_metalabels(self):
        meta_labels = json.load(open(self._pipelines_path, 'r'))
        meta_targets = json.load(open(self._scores_path, 'r'))
        return meta_labels, meta_targets
