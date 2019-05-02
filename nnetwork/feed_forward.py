import numpy as np
from .utils import vectorized_func, sigmoid


def bias(X):
    rows, cols = X.shape
    return np.full((1, cols), 1)


def feed_forward(X, h_weights, o_weights):
    X_biased = np.vstack((X, bias(X)))
    z_h = X_biased @ h_weights
    a_h = vectorized_func(z_h, sigmoid)

    a_h_biased = np.vstack((a_h, bias(a_h)))
    z_o = a_h_biased @ o_weights
    a_o = vectorized_func(z_o, sigmoid)
    return z_h, z_o, a_h, a_o
