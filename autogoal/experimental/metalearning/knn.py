# from sklearn.neighbors import KNeighborsClassifier
from typing import List
from pathlib import Path
from autogoal.experimental.metalearning.datasets import Dataset
from autogoal.experimental.metalearning.metalearner import MetaLearner
import xgboost as xgb


class KNNMetaLearner(MetaLearner):
    def __init__(self, k: int = 5, number_of_results: int = 5):
        """
        k: k parameter in K-means
        number_of_results: number of results to return in each prediction
        """
        super().__init__()
        self.model = xgb.XGBRanker(
            tree_method='gpu_hist',
            booster='gbtree',
            objective='rank:pairwise',
            random_state=42,
            learning_rate=0.1,
            colsample_bytree=0.9,
            eta=0.05,
            max_depth=6,
            n_estimators=110,
            subsample=0.75
        )
        self.samples = None
        self.samples_labels = None
        self.n_results = number_of_results

    def train(self, datasets: List[Dataset]):
        features, labels, targets = self.get_training_samples(datasets)
        self.samples_labels = labels

    def predict(self, dataset: Dataset):
        features = self.preprocess_metafeature(dataset)
        samples, distance = self.model.kneighbors(features, self.n_results)
        # check the samples are sorted by distance
        return [self.samples_labels[i] for i in samples]
