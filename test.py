from autogoal.ml import AutoML
from autogoal.kb import MatrixContinuousDense, VectorCategorical, Supervised, Tensor, Continuous, Dense, Categorical
from autogoal.ml._metalearning import DatasetFeatureLogger

from sklearn.model_selection import train_test_split
from autogoal.search import RichLogger
from autogoal.experimental.metalearning.xgb_ranker import XGBRankerMetaLearner
from autogoal.experimental.metalearning import  DatasetExtractor, DatasetType
from autogoal.datasets import cars
from pathlib import Path
from autogoal.utils import Min



def test_automl(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    # automl = AutoML(
    #         input=(MatrixContinuousDense,
    #                Supervised[VectorCategorical]),
    #         output=VectorCategorical
    # )


    automl = AutoML(
            input=(Tensor[2, Continuous, Dense],
                   Supervised[Tensor[1, Categorical, Dense]]),
            output=Tensor[1, Categorical, Dense],
            evaluation_timeout=5 * Min,
            search_timeout=30 * Min,
            # metalearner=XGBRankerMetaLearner()
    )

    automl.fit(X_train, y_train, logger=RichLogger())

    print(automl.best_pipeline_)
    print(automl.best_score_)

    score = automl.score(X_test, y_test)
    print(f"Score: {score:0.3f}")

    predictions = automl.predict(X_test)

    for sentence, real, predicted in zip(X_test[:10], y_test, predictions):
        print(sentence, "-->", real, "vs", predicted)


def test_datasets(datasets):
    for d in datasets:
        print(d.name)
        X_train, y_train, X_test, y_test = d.load()
        print(f'input_type: {d.input_type}')
        print(f'output_type: {d.output_type}')


if __name__ == '__main__':
    # X, y = cars.load()
    # test_automl(X, y)

    datasets = DatasetExtractor(Path('/home/coder/.autogoal/data/classification/lt 5000')).datasets
    # test_datasets(datasets)
    print(len(datasets))
    learner = XGBRankerMetaLearner()
    # learner.train(datasets)
    learner.meta_train(datasets[:60])
    predictions = learner.test(datasets[60:])
    for (pipeline_info, pipeline_types, _) in predictions:
        model = learner.construct_initial_model(pipeline_info, pipeline_types)
    print(model)
    # learner.evaluate_datasets(datasets[60:])
