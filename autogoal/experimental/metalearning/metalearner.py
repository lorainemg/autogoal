from autogoal.experimental.metalearning.metafeatures import MetaFeatureExtractor
from autogoal.experimental.metalearning.datasets import DatasetExtractor, Dataset
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

    def extract_features(self, datasets: List[Dataset]):
        return [self.metafeature_extractor.extract_features(d.X, d.y) for d in datasets]