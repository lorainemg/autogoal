import functools
import operator
import numpy as np
from scipy import stats

from autogoal.experimental.metalearning.utils import reduce_shape, get_numerical_features
from autogoal.kb import SemanticType
from sklearn.decomposition import PCA
_EXTRACTORS = []


class MetaFeatureExtractor:
    """Extractor of meta features"""
    def __init__(self, features_extractors=None):
        self.feature_extractors = list(features_extractors or _EXTRACTORS)

    def extract_features(self, X, y, dataset):
        features = {}

        # X = convert_to_np_arrays(X)
        # if y is not None:
        #     y = convert_to_np_arrays(y)

        for extractor in self.feature_extractors:
            features.update(**extractor(X, y, dataset=dataset, features=features))

        return features


def convert_to_np_arrays(X):
    """
    Converts the input arrays to dense numpy arrays to allow the methods to work properly
    """
    try:
        X = X.todense()
    except:
        pass
    X = np.array(X)
    if len(X.shape) > 2:
        X = reduce_shape(X)
    return X


# Meta-features

def feature_extractor(func):
    @functools.wraps(func)
    def wrapper(X, y, dataset, **kwargs):
        try:
            result = func(X, y, dataset=dataset, **kwargs)
        except Exception as e:
            print(f'Error in {dataset.name} in {func.__name__} method \n\t{e}\n\n')
            result = None
            # raise

        f_name = func.__name__
        if isinstance(result, tuple):
            # There is more than one result
            feat = {f_name: result[0]}
            for i, res in enumerate(result[1:], 1):
                feat[f'{f_name}_{i}'] = res
        else:
            feat = {f_name: result}

        return feat

    _EXTRACTORS.append(wrapper)
    return wrapper


### Feature extractor methods ###

# Autogoal features
# Features specific to the autogoal characteristics

@feature_extractor
def is_supervised(X, y, **kwargs):
    """Determines if a given dataset is supervised"""
    return y is not None


@feature_extractor
def has_numeric_features(X, y, dataset, **kwargs):
    return any(indicator for indicator in dataset.numerical_indicator)


@feature_extractor
def average_number_of_words(X, y, **kwargs):
    return sum(len(sentence.split(" ")) for sentence in X) / len(X)


@feature_extractor
def has_categorical_features(X, y, dataset, **kwargs):
    return any(not indicator for indicator in dataset.numerical_indicator)


@feature_extractor
def semantic_input_types(X, y, dataset, **kwargs):
    return str(dataset.input_type)


@feature_extractor
def semantic_output_types(X, y, dataset, **kwargs):
    return str(dataset.output_type)


# General features
# General meta-features include general information related to the dataset at hand
# and, to a certain extent, they are conceived to measure the complexity and/or
# the size of the underlying problem.


@feature_extractor
def number_of_samples(X, y, **kwargs):
    """It represents the total number of samples in the dataset"""
    try:
        return X.shape[0]
    except:
        return len(X)


@feature_extractor
def input_dimensionality(X, y, **kwargs):
    """
    It represents the total number of attributes of the dataset
    (i.e. the dimensionality of the input vector.)
    """
    # d = 1
    # for di in X.shape[1:]:
    #     d *= di
    return functools.reduce(operator.mul, X.shape[1:], 1)


@feature_extractor
def output_dimensionality(X, y, **kwargs):
    """
    It represents the total number of output values in the dataset
    (i.e. the dimensionality of the output vector.)
    """
    if y is None:
        return 0
    else:
        return functools.reduce(operator.mul, y.shape[1:], 1)


@feature_extractor
def dataset_dimensionality(X, y, features, **kwargs):
    """
    It represents the ratio between the number of attributes
    and the number of observations constituting the dataset
    """
    try:
        input_dim = features['input_dimensionality']
        num_samp = features['number_of_samples']
        return input_dim / num_samp
    except KeyError:
        return input_dimensionality.__wrapped__(X, y) / number_of_samples.__wrapped__(X, y)


@feature_extractor
def number_of_categorical_features(X, y, dataset, **kwargs):
    # categorical_types = '?bBOU'     # boolean, unsigned and signed bytes, object, unicode_string
    # cat_list = [dtype for dtype in list(X.dtypes) if dtype.kind in categorical_types]
    return sum(not indicator for indicator in dataset.numerical_indicator)


@feature_extractor
def number_of_numerical_features(X, y, dataset, **kwargs):
    return sum(indicator for indicator in dataset.numerical_indicator)


@feature_extractor
def number_of_missing_values(X, y, **kwargs):
    total = np.isnan(X)
    for _ in range(len(X.shape) - 1):
        total = total.sum()
    return int(total)

# Statistical metafeatures
# Statistical metafeatures describe the numerical properties of a distribution of data.
# Can be employed to take into account the number of properties, which enable a learner
# to discriminate the degree of correlation of numerical attributes and estimate their
# distribution


@feature_extractor
def standard_deviation(X, y, dataset, **kwargs):
    """This quantity estimates the dispersion of a random variable"""
    return np.std(X[:, dataset.numerical_indicator])


@feature_extractor
def coefficient_of_variation(X, y, features, dataset, **kwargs):
    """It evaluates the normalization of the standard deviation of a random variable"""
    # var_coefs = np.asarray([x_i.std() / x_i.mean() for x_i in X])
    # return var_coefs.mean()
    return features['standard_deviation'] / np.mean(X[:, dataset.numerical_indicator])


@feature_extractor
def covariance_avg(X, y, dataset, **kwargs):
    """
    As a measure of the covariance of an entire dataset, the average of the covariance
    over all distinct pairs of numerical attributes could be considered.
    """

    return np.cov(X[:, dataset.numerical_indicator], rowvar=False).mean()


@feature_extractor
def linear_corr_coef(X, y, dataset, **kwargs):
    """
    Correlation analysis attempts to measure the strength of a relationship between two
    random variables. It shows the linear association strengths between the random variables
    by means of a single value.
    """
    rho, _ = stats.spearmanr(X[:, dataset.numerical_indicator])
    if isinstance(rho, (float, int)):
        return rho
    return rho.mean()


@feature_extractor
def skewness(X, y, dataset, **kwargs):
    """
    It measures the lack of symmetry in the distribution of a random variable X.
    Negative values indicate data is skewed left, while positive skewness values
    denote data that are skewed right.
    """
    skew = stats.skew(X[:, dataset.numerical_indicator])
    skew = skew.astype('float64')
    return np.mean(skew), np.min(skew), np.max(skew), np.std(skew)


@feature_extractor
def kurtosis(X, y, dataset, **kwargs):
    """
    It measures the peakness in the distribution of a random variable X.
    """
    kurtosis = stats.kurtosis(X[: dataset.numerical_indicator])
    kurtosis = kurtosis.astype('float64')
    return np.mean(kurtosis), np.min(kurtosis), np.max(kurtosis), np.std(kurtosis)


# @feature_extractor
# def variance_fraction_coeff(X, y=None):
#     """
#     This coefficient indicates the relative importance of the largest
#     eigenvalue of the attribute covariance matrix and it measures the
#     representation quality of the first principal component.
#     """
#     # NOTE: This can return a complex number
#     cov_matrix = np.cov(X)
#     largest_eigenval, _ = np.linalg.eig(cov_matrix)
#     return max(largest_eigenval) / cov_matrix.trace()


# Information-Theoretic Meta-Features
# Information-theoretic meta-features are particularly appropriate to describe
# (categorical) attributes, but they also fit continuous (categorical) ones.
# # TODO: A discretization of the input and outputs value should be done


@feature_extractor
def normalized_class_entropy(X, y, **kwargs):
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
def normalized_attr_entropy(X, y, numerical_values, **kwargs):
    """
    The attribute entropy value H(X) of a random variable measures the information
    content related to the values that X may assume.
    """
    attributes = [X[:, j] for j in range(X.shape[1])]
    attr_entropy = np.array([_attr_i_entropy(xi) for xi in attributes])
    return attr_entropy.mean(), np.min(attr_entropy), np.max(attr_entropy), np.std(attr_entropy)


def _attr_i_joint_entropy(x_i, y):
    _, counts = np.unique(list(zip(x_i, y)), return_counts=True)
    return stats.entropy(counts)


# def _attr_i_joint_entropy(x_i, y):
#     H = 0
#     n = len(y)
#     for attr, class_ in product(np.unique(x_i), np.unique(y)):
#         p = len([(a, c) for a, c in zip(x_i, y) if a == attr and c == class_]) / n
#         H += -p*np.log2(p) if p > 0 else 0
#     return H


@feature_extractor
def joint_entropy(X, y, numerical_values, **kwargs):
    """
    It measures the total entropy of the combined system of variables, i.e. the pair
    of variables (C, X), which could be represented by a class variable and one of
    the m discretized inputs attributes, respectively.
    """
    attributes_labels = [(X[:, j], y) for j in range(X.shape[1])]
    joint_entropy = np.array([_attr_i_joint_entropy(attr_i, y) for attr_i, y in attributes_labels])
    return joint_entropy.mean(), np.min(joint_entropy), np.max(joint_entropy), np.std(joint_entropy)


@feature_extractor
def mutual_information(X, y, features, numerical_values, **kwargs):
    """It measures the common information shared between two random variables."""
    result = []
    try:
        class_entropy = features['normalized_class_entropy']
    except KeyError:
        class_entropy = normalized_class_entropy.__wrapped__(X, y)
    for j in range(X.shape[1]):
        attr_entropy = _attr_i_entropy(X[:, j])
        joint_entropy = _attr_i_joint_entropy(X[:, j], y)
        result.append(class_entropy + attr_entropy - joint_entropy)
    return sum(result) / len(result)


@feature_extractor
def equivalent_number_of_attr(X, y, features, **kwargs):
    try:
        return features['normalized_class_entropy'] / features['mutual_information']
    except KeyError:
        return normalized_class_entropy.__wrapped__(X, y) / mutual_information.__wrapped__(X, y)


@feature_extractor
def noise_signal_ratio(X, y, features, **kwargs):
    """It measures the amount of irrelevant information contained in a dataset"""
    try:
        useful_information = features['mutual_information']
        non_useful_information = features['normalized_attr_entropy'] - useful_information
    except KeyError:
        useful_information = mutual_information.__wrapped__(X, y)
        non_useful_information = normalized_attr_entropy.__wrapped__(X, y)[0] - useful_information
    return non_useful_information / useful_information


@feature_extractor
def pca(X, y, dataset, **kwargs):
    pca = PCA()
    pca.fit(X[:, dataset.numerical_indicator])
    first_pc = pca.components_[0]
    return stats.skew(first_pc), stats.kurtosis(first_pc)
