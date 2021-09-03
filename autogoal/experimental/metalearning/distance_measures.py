import numpy as np


def cosine_measure(vect_i, vect_j):
    dot_prod = np.dot(vect_i, vect_j)
    vect_i_l2norm = np.sqrt(np.sum(np.power(vect_i, 2)))
    vect_j_l2norm = np.sqrt(np.sum(np.power(vect_j, 2)))
    return dot_prod / (vect_i_l2norm * vect_j_l2norm)


def l1_distance(vect_i, vect_j):
    return sum([abs(xi - xj) for xi, xj in zip(vect_i, vect_j)])


def l2_distance(vect_i, vect_j):
    return np.sqrt(sum([(xi-xj)**2 for xi, xj in zip(vect_i, vect_j)]))
