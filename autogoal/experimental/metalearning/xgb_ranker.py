# from sklearn.neighbors import KNeighborsClassifier
from typing import List
from autogoal.experimental.metalearning.datasets import Dataset, DatasetType
from autogoal.experimental.metalearning.metalearner import MetaLearner
from itertools import chain
from autogoal.kb import Pipeline


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

    def meta_train(self, datasets: List[Dataset]):
        features, labels, targets, files = self.get_training_samples(datasets)
        self.samples = list(zip(features, targets, labels, files))
        features, grp_info = self.append_features_and_labels(features, labels)
        targets = list(chain.from_iterable(targets))
        self.model.fit(features, targets, group=grp_info)

    def predict(self, dataset: Dataset):
        data_features = self.preprocess_metafeatures(dataset)

        # get the pipelines to test
        datasets = self.get_similar_datasets(data_features, self.cosine_measure)
        pipelines, files = self.get_best_pipelines(datasets, 5, 5)

        features, _ = self.append_features_and_labels([data_features], [pipelines])
        y_hat = self.model.predict(features)

        decode_pipeline = self.decode_pipelines(pipelines)
        pipelines_info = self.get_all_pipeline_info(decode_pipeline, files)
        return pipelines_info, y_hat


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
        datasets = []
        for feat, scores, pipes, file in self.samples:
            sorted_pipes = [p for p, _ in sorted(zip(pipes, scores), key=lambda x: x[1])]
            similarity = similarity_measure(feat, features)
            datasets.append((similarity, sorted_pipes, file))
        # Return the sorted list of the most similar datasets
        return sorted(datasets, key=lambda x: x[0], reverse=True)

    def get_best_pipelines(self, similar_datasets: List, amount_datasets: int, amount_pipelines: int):
        """
        Given the best pipelines to test, this uses a weighted scheme to select between of similar datasets
        and the amount of the best pipelines selected from each dataset
        """
        pipelines = []
        files = []
        for _, pipes, file in similar_datasets[:amount_datasets]:
            pipes = pipes[:amount_pipelines]
            pipelines.extend(pipes)
            files.extend([file]*len(pipes))
        return pipelines, files

    def get_all_pipeline_info(self, pipelines, datasets_path):
        """
        Get the complete info of the pipelines ( to get the hyper-parameters )
        """
        pipeline_info = []
        for pipe, dataset_path in zip(pipelines, datasets_path):
            feature = {}
            self.load_dataset_features(dataset_path, feature)
            metalabels = list(feature.values())[0]['meta_labels']
            pipe = pipe[pipe.nonzero()]
            for pipeline in metalabels['features']:
                algorithms = self.get_pipeline_algorithms(pipeline)
                if len(pipe) == len(algorithms) and all(pipe == algorithms):
                    pipeline_info.append(pipeline)
                    break
            else:
                print(dataset_path)
        return pipeline_info
