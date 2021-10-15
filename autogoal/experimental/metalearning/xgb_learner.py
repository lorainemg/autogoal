# from sklearn.neighbors import KNeighborsClassifier
from typing import List
from autogoal.experimental.metalearning.datasets import Dataset
from autogoal.experimental.metalearning.metalearner import MetaLearner
from autogoal.experimental.metalearning.distance_measures import cosine_measure, l1_distance, l2_distance
from itertools import chain
import pickle

import xgboost as xgb
import numpy as np


class XGBRankerMetaLearner(MetaLearner):
    def __init__(self, features_extractor=None, load=True, number_of_results: int = 15):
        """
        number_of_results: number of results to return in each prediction
        """
        super().__init__(features_extractor, load, learner_name="xgb_metalearner")
        self.n_results = number_of_results

    def _try_to_load_model(self, load):
        if load:
            try:
                self.load_vectors()
                return pickle.load(open(self._model_path, 'rb'))
            except FileNotFoundError:
                load = False
        if not load:
            return xgb.XGBRanker(
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

    def meta_train(self, datasets: List[Dataset], *, save=True):
        features, labels, targets, files = self.get_training_samples(datasets)
        self.samples = list(zip(features, targets, labels, files))
        features, grp_info = self.append_features_and_labels(features, labels)
        targets = list(chain.from_iterable(targets))
        self.model.fit(features, targets, group=grp_info)
        if save:
            pickle.dump(self.model, open(f'{self._model_path}', 'wb'))
            self.save_vectors()

    def predict(self, dataset: Dataset):
        data_features = self.preprocess_metafeatures(dataset)

        # get the pipelines to test
        datasets = self.get_similar_datasets(data_features, l2_distance)

        pipelines, files, _ = self.get_best_pipelines(datasets, self.n_results, self.n_results)

        features, _ = self.append_features_and_labels([data_features], [pipelines])
        y_hat = self.model.predict(features)

        pipelines, files, scores = self._sort_pipelines_by_score(pipelines, files, y_hat)
        pipelines, files, scores = pipelines[:self.n_results], files[:self.n_results], scores[:self.n_results]

        decode_pipeline = self.decode_pipelines(pipelines)
        pipelines_info, pipeline_types = self.get_all_pipeline_info(decode_pipeline, files)
        return pipelines_info, pipeline_types, scores

    @staticmethod
    def _convert_nan_to_zero(vect):
        vect[np.isnan(vect)] = 0
