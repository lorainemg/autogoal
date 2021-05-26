import json
import uuid

from autogoal.search import Logger
from autogoal.utils import nice_repr
from autogoal.experimental.metalearning.metafeature_extractor import _EXTRACTORS


class DatasetFeatureLogger(Logger):
    def __init__(
        self,
        X,
        y=None,
        extractor=None,
        output_file="metalearning.json",
        problem_features=None,
        environment_features=None,
    ):
        self.extractor = extractor or MetaFeatureExtractor()
        self.X = X
        self.y = y
        self.run_id = str(uuid.uuid4())
        self.output_file = output_file
        self.problem_features = problem_features or {}
        self.environment_features = environment_features or {}

    def begin(self, generations, pop_size):
        self.dataset_features_ = self.extractor.extract_features(self.X, self.y)

    def eval_solution(self, solution, fitness):
        if not hasattr(solution, "sampler_"):
            raise ("Cannot log if the underlying algorithm is not PESearch")

        sampler = solution.sampler_

        features = {k: v for k, v in sampler._updates.items() if isinstance(k, str)}
        feature_types = {k: repr(v) for k, v in sampler._model.items() if k in features}

        info = SolutionInfo(
            uuid=self.run_id,
            fitness=fitness,
            problem_features=dict(self.dataset_features_, **self.problem_features),
            environment_features=dict(self.environment_features),
            pipeline_features=features,
            feature_types=feature_types,
        ).to_dict()

        with open(self.output_file, "a") as fp:
            fp.write(json.dumps(info) + "\n")


class MetaFeatureExtractor:
    "Extractor of meta features"
    def __init__(self, features_extractors=None):
        self.feature_extractors = list(features_extractors or _EXTRACTORS)

    def extract_features(self, X, y=None):
        features = {}

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
