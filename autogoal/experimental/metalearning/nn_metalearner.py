from typing import List
from autogoal.experimental.metalearning.datasets import Dataset
from autogoal.experimental.metalearning.metalearner import MetaLearner
from autogoal.experimental.metalearning.distance_measures import cosine_measure, l2_distance, l1_distance


class NNMetaLearner(MetaLearner):
    def __init__(self, features_extractor=None, load=True, number_of_results: int = 15, strategy='aggregated',
                 distance_metric=l2_distance, *, learner_name='nn_metalearner'):
        super().__init__(features_extractor, load, learner_name=learner_name)
        self.n_results = number_of_results
        self.strategy = strategy
        self.distance_metric = distance_metric

    def _try_to_load_model(self, load):
        if load:
            try:
                self.load_vectors()
            except FileNotFoundError:
                pass

    def meta_train(self, datasets: List[Dataset], *, save=True):
        features, labels, targets, files = self.get_training_samples(datasets)
        self.samples = list(zip(features, targets, labels, files))
        # features, _ = self.append_features_and_labels(features, labels)

        if save:
            self.save_vectors()

    def predict(self, dataset: Dataset):
        data_features = self.preprocess_metafeatures(dataset)

        # get the pipelines to test
        if self.strategy == 'aggregated':
            datasets = self.get_similar_datasets(data_features, self.distance_metric, return_similarity=True)
            pipelines, files, scores = self.get_best_pipelines_aggregated_ranking(datasets, self.n_results)
        elif self.strategy == 'simple':
            datasets = self.get_similar_datasets(data_features, self.distance_metric, return_similarity=False)
            pipelines, files, scores = self.get_best_pipelines(datasets, self.n_results, self.n_results)
            pipelines, files, scores = self._sort_pipelines_by_score(pipelines, files, scores)
        else:
            raise Exception('Not a valid strategy')
        pipelines, files, scores = pipelines[:self.n_results], files[:self.n_results], scores[:self.n_results]

        decode_pipeline = self.decode_pipelines(pipelines)
        pipelines_info, pipeline_types = self.get_all_pipeline_info(decode_pipeline, files)
        return pipelines_info, pipeline_types, scores


