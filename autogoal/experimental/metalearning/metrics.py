"""Module to implement the evaluation metrics"""
import math
import functools
from typing import List
from scipy import stats
import numpy as np

_METRICS = []
_CMETRICS = []


def raw_wrapper(func, true_set: List, predicted_set: List, *args):
    try:
        result = func(true_set, predicted_set, *args)
    except:
        result = 0
    return {func.__name__: result}


def metrics_extractor(func):
    wrapper = functools.wraps(func)(functools.partial(raw_wrapper, func))
    _METRICS.append(wrapper)
    return wrapper


def classic_metrics_extractor(func):
    wrapper = functools.wraps(func)(functools.partial(raw_wrapper, func))
    _CMETRICS.append(wrapper)
    return wrapper


@metrics_extractor
def srcc_score(true_set: List[float], predicted_set: List[float]):
    """
    Spearman's Rank Correlation Coefficient: assess how well
    the relationship between the true and prediceted rankings.
    """
    return stats.spearmanr(true_set, predicted_set)[0]
    # n = len(true_set)
    # d2 = [(t-p)**2 for t, p in zip(true_set, predicted_set)]
    # return 1 - 6 * sum(d2) / n * (n**2-1)


@metrics_extractor
def wrc_score(true_set: List[float], predicted_set: List[float]):
    """
    Weighted Rank Correlation: this metrics puts more weight on the top candidates.
    (buscar en el paper `Meta-Learning and the Full Model Selection Problem` referencias de donde se ha usado)
    """
    n = len(true_set)
    d2 = [(t - p) ** 2 for t, p in zip(true_set, predicted_set)]
    to_sum = [di * ((n - t + 1) + (n - p + 1)) for di, t, p in zip(d2, true_set, predicted_set)]
    return 1 - 6 * sum(to_sum) / (n ** 4 + n ** 3 - n ** 2 - n)


@metrics_extractor
def dcg_score(true_set: List[float], predicted_set: List[float], p=None):
    """
    Discounted Cumulative Gain: the premise is that highly relevant documents appearing lower in a
    search result list should be penalized as the graded relevance value is reduced logarithmically
    proportional to the position of the result.

    Sum the true scores ranked in the order induced by the predicted scores.
    """
    p = p or len(true_set) - 1
    sorted_set = sorted(zip(true_set, predicted_set), key=lambda x: x[1], reverse=True)
    predicted = [t for t, _ in sorted_set]
    return sum((2 ** r - 1) / math.log2(i + 1) for i, r in enumerate(predicted[:p], 1))


def idcg_score(true_set: List[float], predicted_set: List[float], p=None):
    """
    Ideal Discounted Cumulative Gain: sorts all relevant documents in the corpus by their relative relevance,
    producing the maximum possible DCG through position p
    """
    p = p or len(true_set) - 1
    predicted = sorted(true_set, reverse=True)
    return sum((2 ** r - 1) / math.log2(i + 1) for i, r in enumerate(predicted[:p], 1))


@metrics_extractor
def ndcg_score(true_set: List[float], predicted_set: List[float], p=None):
    """
    Normalized Discounted Cumulative Gain: s a measure of effectiveness of a search engine algorithm
    or related applications by using a graded relevance scale of items in a result list.
    """
    return dcg_score.__wrapped__(true_set, predicted_set, p) / idcg_score(true_set, predicted_set, p)


def mrr(true_set: List[float], predicted_set: List[float]):
    """
    Mean Reciprocal Rank: is essentially the average of the reciprocal ranks of
    "the first relevant item" for a set of queries Q.
    """
    rr = []  # list of reciprocal ranks
    for t, p in zip(true_set, predicted_set):
        sorted_elements = sorted(enumerate(p), key=lambda x: x[1])
        idx = 0
        for i, value in sorted_elements:
            if i == 0:
                idx = i
                break
        rr.append(1 / idx)
    return 1 / len(true_set) * sum(rr)


# ---------------------------- Classic scores ---------------------------- #


@classic_metrics_extractor
def precision_score(relevant: List, recovered: List) -> float:
    """Precision score is: which of the documents marked as relevant are really relevant"""
    # Recovered relevant
    rr = [d for d in recovered if d in relevant]
    return len(rr) / len(recovered)


@classic_metrics_extractor
def recall_score(relevant: List, recovered: List) -> float:
    """Recall score is: which of the total relevant documents where recovered"""
    # Recovered relevant
    rr = [d for d in recovered if d in relevant]
    return len(rr) / len(relevant)


@classic_metrics_extractor
def fbeta_score(relevant: List, recovered: List, beta: float) -> float:
    """Score that harmonize precision and recall"""
    p = precision_score(relevant, recovered)
    r = recall_score(relevant, recovered)
    try:
        return (1 + beta ** 2) / (1 / p + (beta ** 2) / r)
    except ZeroDivisionError:
        return 0


@classic_metrics_extractor
def f1_score(relevant: List, recovered: List) -> float:
    """Particular case of the fbeta_score"""
    return fbeta_score(relevant, recovered, 1)


@classic_metrics_extractor
def fallout_score(relevant: List, recovered: List, total: int) -> float:
    # Recovered no relevant
    ri = [d for d in recovered if d not in relevant]
    # Total of non relevant documents
    irrelevant = total - len(relevant)
    return len(ri) / irrelevant


# ---------------------------- R scores ---------------------------- #


@classic_metrics_extractor
def r_precision_score(relevant: List, recovered: List, r: int) -> float:
    """Precision for `r` relevant documents"""
    return precision_score(relevant, recovered[:r])


@classic_metrics_extractor
def r_recall_score(relevant: List, recovered: List, r: int) -> float:
    """Precision for `r` relevant documents"""
    return recall_score(relevant, recovered[:r])


@classic_metrics_extractor
def r_f1_score(relevant: List, recovered: List, r: int) -> float:
    """Precision for `r` relevants documents"""
    return f1_score(relevant, recovered[:r])


@classic_metrics_extractor
def r_fallout_score(relevant: List, recovered: List, r: int, total: int) -> float:
    """Precision for `r` relevants documents"""
    return fallout_score(relevant, recovered[:r], total)
