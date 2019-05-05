import numpy as np
from math import exp, log


def vectorized_func(X, func):
    rows, cols = X.shape
    v_func = np.vectorize(func)
    v_values = v_func(X.ravel())
    return v_values.reshape(rows, cols)


def sigmoid(x):
    # if x < 0:
    #     return 1 - (1 / (1 + exp(x)))
    return 1 / (1 + exp(-x))


def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))


def normalize(x, min, max):
    return (x - min) / (max - min)


def normalize_matrix(X, min, max):
    rows, cols = X.shape
    n_func = np.vectorize(normalize)
    return n_func(X.ravel(), min, max).reshape(rows, cols)


def cost(pred, Y):
    rows, cols = Y.shape
    diff_squared = (pred - Y) ** 2
    return 1 / (2 * rows) * np.sum(diff_squared)
