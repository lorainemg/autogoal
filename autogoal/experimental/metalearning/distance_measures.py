import numpy as np


def cosine_measure(vect_i, vect_j):
    dot_prod = np.dot(vect_i, vect_j)
    vect_i_l2norm = np.sqrt(np.sum(np.power(vect_i, 2)))
    vect_j_l2norm = np.sqrt(np.sum(np.power(vect_j, 2)))
    return dot_prod / (vect_i_l2norm * vect_j_l2norm)


def jaccard_distance(self, vect_i, vect_j):
    """Binary distance between 2 documents"""
    intersect = len(vect_i.intersection(vect_j))
    union = len(vect_i.union(vect_j))
    return intersect / union
