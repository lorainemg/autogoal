# from sklearn.neighbors import KNeighborsClassifier
from typing import List
from autogoal.experimental.metalearning.datasets import Dataset, DatasetType
from autogoal.experimental.metalearning.metalearner import MetaLearner
import xgboost as xgb
import numpy as np


class XGBRankerMetaLearner(MetaLearner):
    def __init__(self, number_of_results: int = 5):
        """
        k: k parameter in K-means
        number_of_results: number of results to return in each prediction
        """
        super().__init__()
        self.model = xgb.XGBRanker(
            tree_method='hist',
            booster='gbtree',
            objective='rank:pairwise',
            random_state=42,
            learning_rate=0.1,
            colsample_bytree=0.9,
            eta=0.05,
            max_depth=6,
            n_estimators=110,
            subsample=0.75,
            predictor='cpu_predictor',
        )
        self.samples = None
        self.n_results = number_of_results

    def train(self, dataset_type: DatasetType):
        features, labels, targets = self.get_training_samples(dataset_type)
        self.samples = list(zip(features, labels))
        features = self.append_features_and_labels(features, labels)
        grp_info = self.make_grp_info(features)
        self.model.fit(features, targets, group=grp_info)

    def make_grp_info(self, features):
        _, counts = np.unique(features, axis=0, return_counts=True)
        return counts

    def predict(self, dataset: Dataset):
        data_features = self.preprocess_metafeature(dataset)
        pipelines = self.get_similar_datasets(data_features, self.cosine_measure)
        data_features = self.create_duplicate_data_features(data_features, len(pipelines))
        features = self.append_features_and_labels(data_features, pipelines)
        y_hat = self.model.predict(features)
        sort_for_rank = sorted(zip(y_hat, pipelines), key=lambda x: x[0])
        pipelines = [p for r, p in sort_for_rank]
        return self.decode_pipelines(pipelines)

    @staticmethod
    def cosine_measure(vect_i, vect_j):
        dot_prod = np.dot(vect_i, vect_j)
        vect_i_l2norm = np.sqrt(np.sum(np.power(vect_i, 2)))
        vect_j_l2norm = np.sqrt(np.sum(np.power(vect_j, 2)))
        return dot_prod / (vect_i_l2norm * vect_j_l2norm)

    @staticmethod
    def _convert_nan_to_zero(vect):
        vect[np.isnan(vect)] = 0

    def get_similar_datasets(self, features, similarity_measure) -> List:
        """Get the pipelines of the datasets with similar features of actual dataset"""
        pipelines = []
        for feat, pipes in self.samples:
            similarity = similarity_measure(feat, features)
            pipelines.append((similarity, pipes))
        sorted_by_sim = sorted(pipelines, key=lambda x: x[0], reverse=True)
        best_pipelines = [p for s, p in sorted_by_sim[:10]]
        return best_pipelines
