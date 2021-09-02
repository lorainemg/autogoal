from autogoal.search import Logger
from autogoal.experimental.metalearning.metafeatures import MetaFeatureExtractor
from autogoal.experimental.metalearning.datasets import Dataset, DatasetType
from autogoal.experimental.metalearning.utils import MTL_RESOURCES_PATH
from pathlib import Path
import json


class DatasetFeatureLogger(Logger):
    def __init__(
        self,
        X, y,
        dataset: Dataset,
        output_folder: Path = None,
        extractor=None
    ):
        self.extractor = MetaFeatureExtractor(extractor)
        self.dataset = dataset
        self.X = X
        self.y = y
        self.output_folder = output_folder or Path(MTL_RESOURCES_PATH)
        self.meta_features_ = {}
        self.meta_targets = []
        self._features = []
        self._features_type = []

    def begin(self, generations, pop_size):
        self.meta_features_ = self.extractor.extract_features(self.X, self.y, self.dataset)

    def eval_solution(self, solution, fitness):
        if not hasattr(solution, "sampler_"):
            raise ("Cannot log if the underlying algorithm is not PESearch")
        if fitness == 0:
            return

        features, feature_types = self.extract_pipelines_features(solution.sampler_)
        self._features.append(features)
        self._features_type.append(feature_types)
        self.meta_targets.append(fitness)

    def end(self, best_solution, best_fn):
        meta_labels = {'features': self._features, 'feature_types': self._features_type}
        dataset_features = {
            'meta_features': self.meta_features_,
            'meta_labels': meta_labels,
            'meta_targets': self.meta_targets
        }
        self.save_dataset_features(self.dataset.name, self.output_folder, dataset_features)

    def extract_pipelines_features(self, sampler):
        """Extracts the features of the pipelines in a comprehensive manner"""
        features = {k: v for k, v in sampler._updates.items() if isinstance(k, str)}
        feature_types = {k: repr(v) for k, v in sampler._model.items() if k in features}
        return features, feature_types

    def save_dataset_features(self, name: str, path: Path, features: dict):
        """Saves a dataset features in the expected file path"""
        p = path / f'{name}.json'
        json.dump(features, open(p, 'w+'))
