import numpy as np
from math import exp, log


def vectorized_func(X, func):
    rows, cols = X.shape
    v_func = np.vectorize(func)
    v_values = v_func(X.ravel())
    return v_values.reshape(rows, cols)


def sigmoid(x):
    if x < 0:
        return 1 - (1 / (1 + exp(x)))
    return 1 / (1 + exp(-x))


def sigmoid_d(a):
    return a * (1 - a)


def normalize(x, min, max):
    return (x - min) / (max - min)


def normalize_matrix(X, min, max):
    rows, cols = X.shape
    n_func = np.vectorize(normalize)
    return n_func(X.ravel(), min, max).reshape(rows, cols)


def cross_entropy(y_p, y):
    if y == 1:
        return -log(y_p)
    else:
        return -log(1 - y_p)


def cost(pred, Y):
    ce_func = np.vectorize(cross_entropy)
    ce_results = ce_func(pred, Y)
    return ce_results.ravel().sum() / Y.size


# def cost(pred, Y):
#     rows, cols = Y.shape
#     diff_squared = (pred - Y) ** 2
#     return 1 / (2 * rows) * np.sum(diff_squared)


def score(real_pred, Y):
    diff = real_pred - Y
    return np.count_nonzero(diff == 0) / Y.shape[0]


def confidence_and_prediction(dummy_pred):
    results = []
    for row in dummy_pred:
        max = np.argmax(row)
        results.append((row[max], max))
    return results
