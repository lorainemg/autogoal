from autogoal.search import RichLogger
from autogoal.experimental.metalearning.utils import MTL_RESOURCES_PATH

from pathlib import Path
import math
import uuid
import json


class ResultsLogger(RichLogger):
    def __init__(self, learner_name: str, name: str=""):
        super().__init__()
        self.name: str = name or str(uuid.uuid4())

        self.path = Path(MTL_RESOURCES_PATH) / 'results' / learner_name
        if not self.path.exists():
            self.path.mkdir(parents=True)
        self.path /= f'{self.name}.json'
        
        self.fns: list = []
        self.pipelines: list = []
        self.pipeline_distributions = []
        self.failed_pipelines: int = 0
        self.generations = [0]

    def begin(self, generations, pop_size):
        super().begin(generations, pop_size)

    def eval_solution(self, solution, fitness):
        super().eval_solution(solution, fitness)
        if not hasattr(solution, "sampler_"):
            raise ("Cannot log if the underlying algorithm is not PESearch")

        if fitness in (0, math.inf, -math.inf):
            self.failed_pipelines += 1

        self.fns.append(fitness)
        pipeline = {k: v for k, v in solution.sampler_._updates.items() if isinstance(k, str)}
        pipeline_distribution = {k: repr(v) for k, v in solution.sampler_._model.items() if k in pipeline}

        self.pipelines.append(pipeline)
        self.pipeline_distributions.append(pipeline_distribution)

    def finish_generation(self, fns):
        super(ResultsLogger, self).finish_generation(fns)
        last_idx_gen = self.generations[-1] + len(fns)
        self.generations.append(last_idx_gen)

    def end(self, best_solution, best_fn):
        super().end(best_solution, best_fn)

        # try:
        #     max_idx = self.fns.index(best_fn)
        # except ValueError:
        #     max_idx = -1

        info = {
            # 'max_idx': max_idx,
            'failed_pipelines': self.failed_pipelines,
            'scores': self.fns,
            'pipelines': self.pipelines,
            'pipeline_distributions': self.pipeline_distributions,
            'generations': self.generations,
            'best_fn': best_fn
        }
        # json.dump(info, open(self.path, 'w'))

