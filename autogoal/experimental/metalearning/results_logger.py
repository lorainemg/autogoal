from autogoal.search import RichLogger
from autogoal.experimental.metalearning.utils import MTL_RESOURCES_PATH

from pathlib import Path
import math
import uuid
import json


class ResultsLogger(RichLogger):
    def __init__(self, name: str=""):
        super().__init__()
        self.name: str = name or str(uuid.uuid4())

        self.path = Path(MTL_RESOURCES_PATH) / 'results'
        if not self.path.exists():
            self.path.mkdir()
        self.path /= f'{self.name}.json'

    def begin(self, generations, pop_size):
        super().begin(generations, pop_size)
        self.fns: list = []
        self.pipelines: list = []
        self.failed_pipelines: int = 0

    def eval_solution(self, solution, fitness):
        super().eval_solution(solution, fitness)
        if not hasattr(solution, "sampler_"):
            raise ("Cannot log if the underlying algorithm is not PESearch")

        if fitness in (0, math.inf, -math.inf):
            self.failed_pipelines += 1
            return

        self.fns.append(fitness)
        pipeline = {k: v for k, v in solution.sampler_._updates.items() if isinstance(k, str)}
        self.pipelines.append(pipeline)

    def end(self, best_solution, best_fn):
        super().end(best_solution, best_fn)

        try:
            max_idx = self.fns.index(best_fn)
        except ValueError:
            max_idx = -1

        info = {
            'max_idx': max_idx,
            'failed_pipelines': self.failed_pipelines,
            'scores': self.fns,
            'pipelines': self.pipelines,
            'best_solution': best_solution,
            'best_fn': best_fn
        }
        json.dump(info, open(self.path, 'w'))
