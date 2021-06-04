from autogoal.ml import AutoML
from autogoal.kb import MatrixContinuousDense, VectorCategorical, Supervised, Tensor, Continuous, Dense, Categorical

from sklearn.model_selection import train_test_split
from autogoal.search import RichLogger
from autogoal.experimental.metalearning import KNNMetaLearner, DatasetExtractor
from autogoal.datasets import cars


def test_automl(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    # automl = AutoML(
    #         input=(MatrixContinuousDense,
    #                Supervised[VectorCategorical]),
    #         output=VectorCategorical
    # )

    # automl = AutoML()

    automl = AutoML(
            input=(Tensor[2, Continuous, Dense],
                   Supervised[Tensor[1, Categorical, Dense]]),
            output=Tensor[1, Categorical, Dense]
    )

    automl.fit(X_train, y_train, logger=RichLogger())

    print(automl.best_pipeline_)
    print(automl.best_score_)

    score = automl.score(X_test, y_test)
    print(f"Score: {score:0.3f}")

    predictions = automl.predict(X_test)

    for sentence, real, predicted in zip(X_test[:10], y_test, predictions):
        print(sentence, "-->", real, "vs", predicted)


if __name__ == '__main__':
    # X, y = cars.load()
    # test_automl(X, y)

    datasets = DatasetExtractor().datasets[:1]
    learner = KNNMetaLearner()
    learner.train(datasets)

