from autogoal.ml import AutoML
from autogoal.kb import MatrixContinuousDense, VectorCategorical, Supervised
from autogoal.datasets import (
    abalone,
    cars,
    cifar10,
    dorothea,
    german_credit,
    gisette,
    movie_reviews,
    shuttle,
    wine_quality,
    yeast
)
from autogoal.search import RichLogger
from autogoal.ml._metalearning import DatasetFeatureLogger
from sklearn.model_selection import train_test_split
from autogoal.experimental.metalearning.metafeatures import MetaFeatureExtractor
from pprint import pprint


def test_automl(X, y):
    X_train, y_train, X_test, y_test = train_test_split(X1, y1, test_size=0.15)

    automl = AutoML(
            input=(MatrixContinuousDense,
                   Supervised[VectorCategorical]),
            output=VectorCategorical
    )

    # automl.fit(X_train, y_train, logger=RichLogger())

    print(automl.best_pipeline_)
    print(automl.best_score_)

    score = automl.score(X_test, y_test)
    print(f"Score: {score:0.3f}")

    predictions = automl.predict(X_test)

    for sentence, real, predicted in zip(X_test[:10], y_test, predictions):
        print(sentence, "-->", real, "vs", predicted)


if __name__ == '__main__':
    X1, y1 = cars.load()
    # X2, y2 = abalone.load()
    # X3_train, y3_train, X3_test, y3_test = cifar10.load()
    # X4_train, y4_train, X4_test, y4_test = dorothea.load()
    # X5, y5 = german_credit.load()
    # X6_train, y6_train, X6_test, y6_test = gisette.load()
    # X8, y8 = movie_reviews.load()
    # X9_train, y9_train, X9_test, y9_test = shuttle.load()
    # X10, y10 = wine_quality.load()
    # X11, y11 = yeast.load()

    feat_ext = MetaFeatureExtractor()
    metafeatures = feat_ext.extract_features(X1, y1)
    pprint(metafeatures)

