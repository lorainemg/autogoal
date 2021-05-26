import functools
import operator
import numpy as np
from scipy import stats

_EXTRACTORS = []


def feature_extractor(func):
    @functools.wraps(func)
    def wrapper(X, y=None):
        try:
            result = func(X, y)
        except:
            result = None
            # raise

        return {func.__name__: result}

    _EXTRACTORS.append(wrapper)
    return wrapper


### Feature extractor methods ###

@feature_extractor
def is_supervised(X, y=None):
    """Determines if a given dataset is supervised"""
    return y is not None


@feature_extractor
def has_numeric_features(X, y=None):
    return any([xi for xi in X[0] if isinstance(xi, (float, int))])


@feature_extractor
def average_number_of_words(X, y=None):
    return sum(len(sentence.split(" ")) for sentence in X) / len(X)


@feature_extractor
def has_text_features(X, y=None):
    return isinstance(X[0], str)


"""
General features
General meta-features include general information related to the dataset at hand
and, to a certain extent, they are conceived to measure the complexity and/or
the size of the underlying problem.
"""


@feature_extractor
def number_of_samples(X, y=None):
    """It represents the total number of samples in the dataset"""
    try:
        return X.shape[0]
    except:
        return len(X)


@feature_extractor
def input_dimensionality(X, y=None):
    """
    It represents the total number of attributes of the dataset
    (i.e. the dimensionality of the input vector.)
    """
    # d = 1
    # for di in X.shape[1:]:
    #     d *= di
    return functools.reduce(operator.mul, X.shape[1:], 1)


@feature_extractor
def output_dimensionality(X, y=None):
    """
    It represents the total number of output values in the dataset
    (i.e. the dimensionality of the output vector.)
    """
    if y is None:
        return 0
    else:
        return functools.reduce(operator.mul, y.shape[1:], 1)


@feature_extractor
def dataset_dimensionality(X, y=None):
    """
    It represents the ratio between the number of attributes
    and the number of observations constituting the dataset
    """
    return input_dimensionality.__wrapped__(X, y) / number_of_samples.__wrapped__(X, y)


"""
Statistical metafeatures
Statistical metafeatures describe the numerical properties of a distribution of data.
Can be employed to take into account the number of properties, which enable a learner
to discriminate the degree of correlation of numerical attributes and estimate their
distribution
"""


@feature_extractor
def standard_deviation(X, y=None):
    """This quantity estimates the dispersion of a random variable"""
    return X.std()


@feature_extractor
def coefficient_of_variation(X, y=None):
    """It evaluates the normalization of the standard deviation of a random variable"""
    # var_coefs = np.asarray([x_i.std() / x_i.mean() for x_i in X])
    # return var_coefs.mean()
    return X.std() / X.mean()


# @feature_extractor
# def covariance(X, y=None):
#     """The covariance extends the variance concept to the dimensional case"""
#     return X.cov()
#

@feature_extractor
def covariance_avg(X, y=None):
    """
    As a measure of the covariance of an entire dataset, the average of the covariance
    over all distinct pairs of numerical attributes could be considered.
    """
    return X.cov().mean()


@feature_extractor
def linear_corr_coef(X, y=None):
    """
    Correlation analysis attempts to measure the strength of a relationship between two
    random variables. It shows the linear association strengths between the random variables
    by means of a single value.
    """
    rho, _ = stats.spearmanr(X)
    return rho


@feature_extractor
def skewness(X, y=None):
    """
    It measures the lack of symmetry in the distribution of a random variable X.
    Negative values indicate data is skewed left, while positive skewness values
    denote data that are skewed right.
    """
    return stats.skew(X).mean()


@feature_extractor
def kurtosis(X, y=None):
    """
    It measures the peakdness in the distribution of a random variable X.
    """
    return stats.kurtosis(X).mean()


@feature_extractor
def variance_fraction_coeff(X, y=None):
    """
    This coefficient indicates the relative importance of the largest
    eigenvalue of the attribute covariance matrix and it measures the
    representation quality of the first principal component.
    """
    cov_matrix = X.cov()
    largest_eigenval, _ = np.linalg.eig(cov_matrix)
    return max(largest_eigenval) / cov_matrix.trace()


"""
Information-Theoretic Meta-Features
Information-theoretic meta-features are particularly appropriate to describe
(categorical) attributes, but they also fit continuous (categorical) ones.
# TODO: A discretization of the input and outputs value should be done
"""


@feature_extractor
def normalized_class_entropy(X, y=None):
    """
    The entropy value H(C) of a class variable C indicates
    how much information is necessary to specify one class.
    """
    _, counts = np.unique(y, return_counts=True)
    return stats.entropy(counts) / np.log2(len(counts))


def _attr_i_entropy(x_i):
    _, counts = np.unique(x_i, return_counts=True)
    return stats.entropy(counts) / np.log2(len(counts))


@feature_extractor
def normalized_attr_entropy(X, y=None):
    """
    The attribute entropy value H(X) of a random variable measures the information
    content related to the values that X may assume.
    """
    attributes = [X[:, j] for j in X.shape[1]]
    return np.array([_attr_i_entropy(xi) for xi in attributes]).mean()


def _attr_i_joint_entropy(x_i, y):
    _, counts = np.unique(zip(x_i, y), return_counts=True)
    return stats.entropy(counts)


# def _attr_i_joint_entropy(x_i, y):
#     H = 0
#     n = len(y)
#     for attr, class_ in product(np.unique(x_i), np.unique(y)):
#         p = len([(a, c) for a, c in zip(x_i, y) if a == attr and c == class_]) / n
#         H += -p*np.log2(p) if p > 0 else 0
#     return H


@feature_extractor
def joint_entropy(X, y=None):
    """
    It measures the total entropy of the combined system of variables, i.e. the pair
    of variables (C, X), which could be represented by a class variable and one of
    the m discretized inputs attributes, respectively.
    """
    attributes_labels = [(X[:, j], y) for j in X.shape[1]]
    return np.array([_attr_i_joint_entropy(attr_i, y) for attr_i, y in attributes_labels]).mean()


@feature_extractor
def mutual_information(X, y=None):
    """It measures the common infomation shared between two random variables."""
    result = []
    class_entropy = normalized_class_entropy.__wrapped__(X, y)
    for j in range(X.shape[1]):
        attr_entropy = _attr_i_entropy(X[:, j])
        joint_entropy = _attr_i_joint_entropy(X[:, j], y)
        result.append(class_entropy + attr_entropy - joint_entropy)
    return sum(result) / len(result)


@feature_extractor
def equivalent_number_of_attr(X, y=None):
    return normalized_class_entropy.__wrapped__(X, y) / mutual_information.__wrapped__(X, y)


@feature_extractor
def noise_signal_ratio(X, y=None):
    """It measures the amount of irrelevant information contained in a dataset"""
    useful_information = mutual_information.__wrapped__(X, y)
    non_useful_information = normalized_attr_entropy.__wrapped__(X, y) - mutual_information.__wrapped__(X, y)
    return non_useful_information / useful_information
