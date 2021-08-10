from autogoal.experimental.metalearning.metafeatures import MetaFeatureExtractor
from autogoal.experimental.metalearning.datasets import Dataset, DatasetType
from autogoal.experimental.metalearning.utils import pad_arrays, fix_indef_values
from autogoal.ml import AutoML
from autogoal.search import RichLogger
from autogoal.utils import Hour, Min

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from itertools import chain
from pathlib import Path
from typing import List, Tuple
import numpy as np
import json
import re
from os import mkdir

class MetaLearner:
    def __init__(self, k=5, features_extractor=None):
        self.k = k  # the numbers of possible algorithms to predict
        self.meta_feature_extractor = MetaFeatureExtractor(features_extractor)
        self._vectorizer = DictVectorizer()
        self._features_path = Path('autogoal/experimental/metalearning/resources/')
        self._pipelines_encoder = LabelEncoder()

    def train(self, dataset_type: DatasetType):
        raise NotImplementedError

    def predict(self, dataset: Dataset):
        raise NotImplementedError

    def test(self, datasets: List[Dataset]):
        return [self.predict(dataset) for dataset in datasets]

    def extract_metafeatures(self, datasets: List[Dataset], visited_datasets=None):
        """Extracts the features of the datasets"""
        visited_datasets = visited_datasets if visited_datasets is not None else {}
        meta_features = {}
        for d in datasets:
            if d.name in visited_datasets:
                continue
            meta_features[d.name] = self._extract_metafeatures(d)
        return meta_features

    def _extract_metafeatures(self, dataset: Dataset):
        X, y, _, _ = dataset.load()
        return self.meta_feature_extractor.extract_features(X, y, dataset)

    def extract_metatargets(self, datasets: List[Dataset], visited_datasets=None, algorithms=None):
        """
        Extracts the features of the solution.
        For this, a list of algorithms is trained with the datasets.

        Returns a list of the labels with the best pipelines and
        a list of metatargets (features of the training process)
        """
        visited_datasets = {} if visited_datasets is None else visited_datasets
        meta_labels = {}
        meta_targets = {}
        for d in datasets:
            if d.name in visited_datasets:
                continue
            meta_label, meta_target = self._extract_metatargets(d, algorithms)
            meta_labels[d.name] = meta_label
            meta_targets[d.name] = meta_target
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
            evaluation_timeout=1 * Min,
            search_timeout=3 * Min,
            algorithms_list=True
        )
        try:
            fix_indef_values(X_train)
            pipelines, score = automl.fit(X_train, y_train, logger=RichLogger())

            fix_indef_values(X_test)
            # score = automl.score(X_test, y_test)
            # missing more metatargets, maybe training time
            features, features_types = self.extract_pipelines_features(pipelines)
            dict_ = {'features': features, 'features_types': features_types}
            return dict_, score
        except Exception as e:
            print(f'Error {dataset.name}: {e}')
            with open('errors.txt', 'w+') as f:
                f.write(dataset.name)
            return None, None

    def extract_pipelines_features(self, solution):
        """Extracts the features of the pipelines in a comprehensive manner"""
        features = []
        feature_types = []
        # A list of the best solutions is expected
        if not isinstance(solution, list):
            solution = [solution]
        for sol in solution:
            feat, feat_types = self._extract_solution_pipeline(sol)
            features.append(feat)
            feature_types.append(feat_types)
        return features, feature_types

    def _extract_solution_pipeline(self, solution):
        sampler = solution.sampler_
        features = {k: v for k, v in sampler._updates.items() if isinstance(k, str)}
        feature_types = {k: repr(v) for k, v in sampler._model.items() if k in features}
        return features, feature_types

    def save_training_samples(self, datasets: List[Dataset]):
        """Save all datasets features"""
        # Extracts meta_features, a processing of the dataset is done
        feat_path = self.get_features_path(datasets[0].type)

        visited_datasets = self.get_dataset_in_path(feat_path)
        meta_features = self.extract_metafeatures(datasets, visited_datasets)

        # Extracts meta_labels (pipelines) and meta_targets (scores)
        # For this, training is done across all datasets.
        meta_labels, meta_targets = self.extract_metatargets(datasets, visited_datasets)

        # saves the meta_features
        self.save_training_features(meta_features, meta_labels, meta_targets, feat_path)
        return feat_path

    def get_training_samples(self, dataset_type: DatasetType):
        """
        Returns all the features vectorized and the labels.
        """
        path = self.get_features_path(dataset_type)
        features = self.load_training_features(path)

        meta_features, meta_labels, meta_targets = self.separate_features(features)
        # Preprocess meta_labels and meta_features to obtain a vector-like meta_features
        meta_features = self.preprocess_datasets(meta_features)
        meta_labels = self.preprocess_pipelines(meta_labels)
        return meta_features, meta_labels, meta_targets

    def get_features_path(self, dataset_type: DatasetType):
        """
        Gets the real path of the features depending in the type of the dataset.
        """
        type_ = re.match('DatasetType.(\w+)', str(dataset_type))
        path = '' if type_ is None else type_.group(1).capitalize()
        features_path = self._features_path / path
        if not features_path.exists():
            mkdir(features_path)
        return features_path

    def separate_features(self, features: dict):
        meta_features = []
        meta_labels = []
        meta_targets = []
        for feat in features.values():
            meta_features.append(feat['meta_features'])
            meta_labels.append(feat['meta_labels'])
            meta_targets.append(feat['meta_targets'])
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
        fix_indef_values(vect)
        return vect

    def preprocess_pipelines(self, meta_labels):
        pipelines = []
        max_len = 0
        for meta_label in meta_labels:
            features = meta_label['features']
            pipeline = []
            for feat in features:
                for algorithm, param in feat.items():
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

    def preprocess_metafeatures(self, dataset: Dataset):
        meta_feature = self._extract_metafeatures(dataset)
        vect = np.array(self._vectorizer.transform([meta_feature]).todense())[0]
        fix_indef_values(vect)
        return vect

    def decode_pipelines(self, pipelines: List[List[int]]) -> List[List[str]]:
        return [self._pipelines_encoder.inverse_transform(p) for p in pipelines]

    def save_dataset_features(self, name: str, path: Path, features: dict):
        """Saves a dataset features in the expected file path"""
        p = path / f'{name}.json'
        json.dump(features, open(p, 'w+'))

    def load_dataset_features(self, filepath: Path, features: dict):
        """Loads a dataset feature in the expected file path"""
        name = filepath.name[:-5]
        features[name] = json.load(open(filepath, 'r'))

    def load_training_features(self, path: Path) -> dict:
        """Load the json with the features information in the expected format into dict"""
        meta_features = {}
        for file in path.glob('*.json'):
            self.load_dataset_features(file, meta_features)
        return meta_features

    def save_training_features(self, meta_features: dict, meta_labels: dict, meta_targets: dict, path: Path):
        """Saves the training features to json"""
        for name in meta_features.keys():
            dataset_features = {
                'meta_features': meta_features[name],
                'meta_labels': meta_labels[name],
                'meta_targets': meta_targets[name]
            }
            self.save_dataset_features(name, path, dataset_features)

    def get_dataset_in_path(self, path: Path):
        """
        Gets all the dataset with stored information in a specific path.
        This is used to check which datasets has features extracted.
        """
        return set(file.name[:-5] for file in path.glob('*.json'))
