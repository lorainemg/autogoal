from autogoal.ml import AutoML
from autogoal.kb import MatrixContinuousDense, VectorCategorical, Supervised

from sklearn.model_selection import train_test_split



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
    dateset_ext = DatasetExtractor()


