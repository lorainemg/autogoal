from autogoal.experimental.metalearning.datasets import Dataset, DatasetType
from autogoal.experimental.metalearning.utils import pad_arrays, fix_indef_values, train_test_split, MTL_RESOURCES_PATH
from autogoal.experimental.metalearning.metafeatures import MetaFeatureExtractor
from autogoal.experimental.metalearning.datasets_logger import DatasetFeatureLogger
from autogoal.ml.metrics import accuracy
from autogoal.experimental.metalearning.metrics import _METRICS
from autogoal.ml import AutoML
from autogoal.utils import Hour, Min
from autogoal.contrib import find_classes
from autogoal.kb import Pipeline
from autogoal import grammar
from autogoal.sampling import MeanDevParam, UnormalizedWeightParam, DistributionParam, WeightParam
from autogoal.sampling import update_model, merge_updates

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from numpy import mean
from itertools import chain
from pathlib import Path
from typing import List, Tuple, Dict

import uuid
import numpy as np
import inspect
import json
import re
from os import mkdir


class MetaLearner:
    def __init__(self, k=5, features_extractor=None):
        self.k = k  # the numbers of possible algorithms to predict
        self.meta_feature_extractor = MetaFeatureExtractor(features_extractor)
        self._vectorizer = DictVectorizer()

        self._resources_path = Path(MTL_RESOURCES_PATH) / 'datasets_info'
        if not self._resources_path.exists():
            self._resources_path.mkdir()

        self._results_path = self._resources_path / 'results'
        if not self._results_path.exists():
            self._results_path.mkdir()
            
        self._results_path.mkdir(exist_ok=True)
        self._pipelines_encoder = LabelEncoder()

    def meta_train(self, dataset_type: DatasetType):
        raise NotImplementedError

    def predict(self, dataset: Dataset):
        raise NotImplementedError

    def test(self, datasets: List[Dataset]):
        return [self.predict(dataset) for dataset in datasets]

    def train(self, datasets: List[Dataset], algorithms=None):
        """
            Hay 3 formas posibles de entrenar:
            1. Entrenar con todos los algoritmos con sus parámetros por defecto
            2. Entrenar con todoos los posibles parámetros a maximizar
            3. Entrenar usando automl para buscar el mejor pipeline.
            Esta versión usará la 3ra opción.
        """
        for dataset in datasets:
            self.train_dataset(dataset, algorithms)

    def train_dataset(self, dataset: Dataset, algorithms=None):
        """Trains a single dataset, storing information about its features"""
        feat_path = self.get_features_path(dataset.type)
        visited_datasets = self.get_datasets_in_path(feat_path)
        if dataset.name in visited_datasets:
            return
        X, y = dataset.load()
        automl = AutoML(
            input=dataset.input_type,
            output=dataset.output_type,
            registry=algorithms,
            evaluation_timeout=5 * Min,
            search_timeout=30 * Min
        )
        try:
            fix_indef_values(X)
            folder = self.get_features_path(dataset.type)
            automl.fit(X, y, logger=DatasetFeatureLogger(X, y, dataset, folder))
        except Exception as e:
            print(f'Error {dataset.name}: {e}')
            with open('errors.txt', 'w+') as f:
                f.write(dataset.name)


    def get_training_samples(self, datasets: List[Dataset]):
        """
        Returns all the features vectorized and the labels and the filenames of the datasets processed.
        """
        # path = self.get_features_path(dataset_type)
        # features, files = self.load_training_features(path)
        features, files = self.load_datasets(datasets)

        meta_features, meta_labels, meta_targets = self.separate_features(features)
        # Preprocess meta_labels and meta_features to obtain a vector-like meta_features
        meta_features = self.preprocess_datasets(meta_features)
        meta_labels = [meta_label['features'] for meta_label in meta_labels]
        meta_labels = self.preprocess_pipelines(meta_labels)
        return meta_features, meta_labels, meta_targets, files

    def get_features_path(self, dataset_type: DatasetType):
        """
        Gets the real path of the features depending in the type of the dataset.
        """
        type_ = re.match('DatasetType.(\w+)', str(dataset_type))
        path = '' if type_ is None else type_.group(1).capitalize()
        features_path = self._resources_path / path
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

        Also returns (if asked) a list of the groups marked by the dataset
        """
        features = []
        grp_info = []
        for i in range(len(meta_labels)):
            dataset_feat = meta_features[i]
            dataset_labels = meta_labels[i]

            grp_size = len(dataset_labels)     # number of times the dataset label is repeated (makes the group info)
            duplicate_feat = self.create_duplicate_data_features(dataset_feat, grp_size)
            for j in range(grp_size):
                duplicate_feat[j].extend(dataset_labels[j])

            features.extend(duplicate_feat)
            grp_info.append(grp_size)
        return np.array(features), grp_info

    def create_duplicate_data_features(self, data_features, n):
        """
        Repeats n times the data_features to obtain various instances of the same list.

        Every instance of this features will be joined with a different pipeline.
        """
        return [list(data_features) for _ in range(n)]

    def preprocess_datasets(self, meta_features):
        self._vectorizer.fit(meta_features)
        vect = np.array(self._vectorizer.transform(meta_features).todense())
        fix_indef_values(vect)
        return vect

    def preprocess_pipelines(self, meta_labels):
        pipelines = []
        max_len = 0
        for features in meta_labels:
            # features = meta_label['features']
            dataset_pipelines = []
            for feat in features:
                pipeline = self.get_pipeline_algorithms(feat)
                max_len = max(max_len, len(pipeline))
                dataset_pipelines.append(pipeline)
            pipelines.append(dataset_pipelines)

        # padds the pipelines so every pipeline has the same length
        padded_pipelines = []
        for dataset_pipeline in pipelines:
            pipeline = np.array([pad_arrays(pipeline, max_len) for pipeline in dataset_pipeline])
            padded_pipelines.append(pipeline)

        # Encodes the pipelines names
        self._pipelines_encoder.fit(list(chain.from_iterable(chain.from_iterable(padded_pipelines))))
        # Transforms the pipelines
        return [[self._pipelines_encoder.transform(p) for p in pipelines] for pipelines in padded_pipelines]

    def get_pipeline_algorithms(self, features):
        """
        Gets the algorithm list saved in the features.
        """
        pipeline = []
        for algorithm, param in features.items():
            if algorithm == 'End':
                break
            # First position in param is the amount of times the algorithm is applied (I think)
            for i in range(param[0]):
                pipeline.append(algorithm)
        return pipeline

    def preprocess_metafeatures(self, dataset: Dataset):
        """
        Method for the predict method to extract only the meta-features of a dataset.
        """
        features = self.load_dataset_feature(dataset)
        if features is None:
            # if dataset was not found train to load the features
            X, y = dataset.load()
            meta_feature = MetaFeatureExtractor().extract_features(X, y, dataset)
            # self.train_dataset(dataset)
            # features = self.load_dataset_feature(dataset)
        else:
            meta_feature = features['meta_features']

        # preprocess the dataset features
        vect = np.array(self._vectorizer.transform([meta_feature]).todense())[0]
        fix_indef_values(vect)
        return vect

    def decode_pipelines(self, pipelines: List[List[int]]) -> List[List[str]]:
        return [self._pipelines_encoder.inverse_transform(p) for p in pipelines]

    def load_dataset_features(self, filepath: Path, features: dict = None):
        """Loads a dataset feature in the expected file path"""
        if features is None:
            features = {}
        name = filepath.name[:-5]
        features[name] = json.load(open(filepath, 'r'))

    def load_training_features(self, path: Path) -> Tuple[dict, List[Path]]:
        """
        Given a path with the information of the datasets stored in a json file
        loads the features information in the expected format into dict
        """
        meta_features = {}
        files = []
        for file in path.glob('*.json'):
            self.load_dataset_features(file, meta_features)
            files.append(file)
        return meta_features, files

    def load_datasets(self, datasets: List[Dataset]) -> Tuple[dict, List[Path]]:
        """
        Given a list of datasets loads the features information in the expected format into dict
        """
        meta_features = {}
        datasets_names = [dataset.name for dataset in datasets]
        files = []
        path = self.get_features_path(datasets[0].type)
        for file in path.glob('*.json'):
            if file.name[:-5] in datasets_names:
                self.load_dataset_features(file, meta_features)
                files.append(file)
        return meta_features, files

    def get_datasets_in_path(self, path: Path):
        """
        Gets all the dataset with stored information in a specific path.
        This is used to check which datasets has features extracted.
        """
        return set(file.name[:-5] for file in path.glob('*.json'))

    def load_dataset_feature(self, dataset: Dataset):
        """Tries to load the features of a dataset if exists"""
        feat_path = self.get_features_path(dataset.type)
        try:
            return json.load(open(feat_path / f'{dataset.name}.json', 'r'))
        except FileNotFoundError:
            return None

    def construct_pipelines(self, pipelines: List[dict], input_type) -> List[Pipeline]:
        """
        Given the list of pipelines information returns a list of built pipelines.
        """
        models = find_classes()
        categorical_values = {x.__name__: self.get_categorical_values(x) for x in models}

        names_alg = {x.__name__: x for x in models}
        alg_re = re.compile('([^_]+)(_(.*))?')
        builded_pipelines = []
        for pipeline in pipelines:
            algorithms = {}
            for key, value in pipeline.items():
                match = alg_re.match(key)
                algorithm = match.group(1)
                param = match.group(3)
                if algorithm == 'End' or algorithm == 'production':
                    continue
                if param is None:
                    algorithms[algorithm] = {}
                else:
                    try:
                        value = categorical_values[algorithm][param][value[0]]
                    except KeyError:
                        value = value[0]
                    algorithms[algorithm][param] = value
            algorithms_list = []
            for algorithm, param in algorithms.items():
                alg = names_alg[algorithm]
                alg = alg(**param)
                algorithms_list.append(alg)
            builded_pipelines.append(Pipeline(algorithms_list, input_type))
        return builded_pipelines

    def get_categorical_values(self, cls):
        """Gets the categorical values of the signature of a function."""
        categorical_values = {}
        # Get the signature
        if inspect.isclass(cls):
            if getattr(cls, "get_inner_signature", None):
                signature = cls.get_inner_signature()
            else:
                signature = inspect.signature(cls.__init__)
        elif inspect.isfunction(cls):
            signature = inspect.signature(cls)
        else:
            raise ValueError("Unable to obtain signature for %r" % cls)

        for param_name, param_obj in signature.parameters.items():
            if param_name in ["self", "args", "kwargs"]:
                continue

            annotation_cls = param_obj.annotation
            if isinstance(annotation_cls, grammar.CategoricalValue):
                categorical_values[param_obj.name] = annotation_cls.options
        return categorical_values

    def score_pipelines(self, X, y, pipelines: List[Pipeline], cross_validation_steps: int = 3):
        """
        Gets the predicted score of the list of pipelines.
        Uses cross validation."""
        scores = {i: [] for i in range(len(pipelines))}
        for _ in range(cross_validation_steps):
            for i, pipeline in enumerate(pipelines):
                try:
                    scr = self.run_pipeline(X, y, pipeline, accuracy)
                    scores[i].append(scr)
                except Exception as e:
                    print('------------------------------------------')
                    print(pipeline)
                    print(e)
                    print('------------------------------------------')
        return self.average_score(scores)

    def run_pipeline(self, X, y, pipeline: Pipeline, score_metric):
        """
        Runs a pipeline given a dataset.
        """
        X_train, y_train, X_test, y_test = train_test_split(X, y)
        pipeline.send("train")
        pipeline.run(X_train, y_train)
        pipeline.send("eval")
        y_pred = pipeline.run(X_test, None)
        return score_metric(y_test, y_pred)

    def get_gold_pred(self, dataset: Dataset):
        """
        Gets golden predictions given the dataset file.
        """
        features = self.load_dataset_feature(dataset)
        if features is None:
            # If dataset was not found trains it to load the features
            # in practice, it should not happen
            self.train_dataset(dataset)
            features = self.load_dataset_feature(dataset)

        # Sort the pipelines by score
        pipe_features = features['meta_labels']['features']
        scores = features['meta_targets']
        score_pipes_by_score = sorted(zip(pipe_features, scores), key=lambda x: x[1], reverse=True)
        pipe_features = [p for p, _ in score_pipes_by_score]
        scores = [s for _, s in score_pipes_by_score]

        return pipe_features, scores

    def average_score(self, score: dict) -> List[float]:
        """Returns the average of the score for every entry"""
        return [mean(values) if values else 0 for values in score.values()]

    def evaluate(self, X, y, pred: List[float], pipelines: List[Pipeline], metrics: List=None) -> Dict[str, float]:
        """Evaluates the proposal given dataset based in a metric"""
        metrics = metrics or _METRICS
        target = self.score_pipelines(X, y, pipelines, 1)
        score = {}
        for metric in metrics:
            score.update(**metric(target, pred))
        return score

    def evaluate_datasets(self, datasets: List[Dataset], save=True, metric=None) -> Dict[str, Dict[str, float]]:
        """Evaluates the proposal given a list of datasets given a metrics"""
        predictions = [self.predict(dataset) for dataset in datasets]
        scores = {}
        for dataset, (pipelines_info, _, y_hat) in zip(datasets, predictions):
            X, y = dataset.load()
            pipelines = self.construct_pipelines(pipelines_info, dataset.input_type)
            scores[dataset.name] = self.evaluate(X, y, y_hat, pipelines, metric)
        scores['global'] = self.calculate_global_score(scores)

        if save:
            json.dump(scores, open(self._results_path / str(uuid.uuid4()), 'w+'))
        return scores

    def calculate_global_score(self, scores: Dict[str, Dict[str, float]]):
        """Calculate average score of all datasets"""
        metrics = {}
        for dataset_score in scores.values():
            for metric, score in dataset_score.items():
                try:
                    metrics[metric].append(score)
                except KeyError:
                    metrics[metric] = [score]
        for metric, score in metrics.items():
            metrics[metric] = mean(score)
        return metrics

    def parse_features_type(self, features_types: List[Dict[str, str]]):
        """
        Parse feature types (models) written as strings and converts them into objects.
        """
        float_re = re.compile(r'(-?(\d*\.\d+|\d+))')
        params = []
        for feature_models in features_types:
            models_params = {handle: self.parse_param(param, float_re) for handle, param in feature_models.items()}
            params.append(models_params)
        return params

    def parse_param(self, param, regex):
        """Helper method that parse the params and converts them into objects"""
        values = [float(number) for number, _ in regex.findall(param)]
        p = None
        try:
            if param.startswith('UnormalizedWeightParam'):
                # value = float_re.findall(param)[0]
                p = UnormalizedWeightParam(*values)
            elif param.startswith('WeightParam'):
                # value = float_re.findall(param)[0]
                p = WeightParam(*values)
            elif param.startswith('MeanDevParam'):
                p = MeanDevParam(*values[:2], initial_params=values[2:])
            elif param.startswith('DistributionParam'):
                p = DistributionParam(values)
        except TypeError as e:
            print(e)
            p = None
        return p

    def construct_initial_model(self, pipeline_updates, pipeline_models):
        """Merges all previous experience to create a initial model"""
        merged_updates = merge_updates(*pipeline_updates)
        model = {}
        for pipeline_model in pipeline_models:
            model.update(pipeline_model)
        return update_model(model, merged_updates)

    def create_initial_set(self, dataset: Dataset):
        pipeline_updates, pipeline_model, _ = self.predict(dataset)
        return self.construct_initial_model(pipeline_updates, pipeline_model)
