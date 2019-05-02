import numpy as np
from math import exp, log


def sigmoid(x):
    if x < 0:
        return 1 - 1 / (1 + exp(x))
    return 1 / (1 + exp(-x))


def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))


def vectorized_func(X, func):
    rows, cols = X.shape
    v_func = np.vectorize(func)
    v_values = v_func(X.ravel())
    return v_values.reshape(rows, cols)


def cross_entropy_loss(pred, Y):
    return -1 / Y.size * (np.sum(Y * np.log(pred)))
