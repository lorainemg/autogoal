from autogoal.utils import nice_repr
from autogoal.experimental.metalearning.metafeature_extractor import _EXTRACTORS
from autogoal.experimental.metalearning.utils import reduce_shape


class MetaFeatureExtractor:
    """Extractor of meta features"""
    def __init__(self, features_extractors=None):
        self.feature_extractors = list(features_extractors or _EXTRACTORS)

    def extract_features(self, X, y=None):
        features = {}

        if len(X.shape) > 2:
            X = reduce_shape(X)
        if y is not None and len(y.shape) > 2:
            y = reduce_shape(y)

        for extractor in self.feature_extractors:
            features.update(**extractor(X, y))

        return features

@nice_repr
class SolutionInfo:
    def __init__(
        self,
        uuid: str,
        problem_features: dict,
        pipeline_features: dict,
        environment_features: dict,
        feature_types: dict,
        fitness: float,
    ):
        self.problem_features = problem_features
        self.pipeline_features = pipeline_features
        self.environment_features = environment_features
        self.feature_types = feature_types
        self.fitness = fitness
        self.uuid = uuid

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(d):
        return SolutionInfo(**d)
