import numpy as np
from .utils import vectorized_func, sigmoid


def bias(X):
    rows, cols = X.shape
    return np.full((rows, 1), 1)


def feed_forward(X, weights):
    layers = len(weights)

    # a = np.hstack((bias(X), X))
    # z_h = X_biased @ weights[0]
    # a_h = vectorized_func(z_h, sigmoid)

    a_l = X
    A = []
    for i in range(0, layers):
        a_l = np.hstack((bias(a_l), a_l))
        A.append(a_l)
        z_l = a_l @ weights[i]
        a_l = vectorized_func(z_l, sigmoid)

    A.append(a_l)

    return A
