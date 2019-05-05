import numpy as np
from .utils import vectorized_func, sigmoid


def bias(X):
    rows, cols = X.shape
    return np.full((1, cols), 1)


def feed_forward(X, weights):
    layers = len(weights) + 1

    X_biased = np.vstack((X, bias(X)))
    z_h1 = X_biased @ weights[0]
    a_h1 = vectorized_func(z_h1, sigmoid)

    Z = [z_h1]
    A = [a_h1]

    for i in range(0, layers - 2):
        a_h_biased = np.vstack((A[i], bias(A[i])))
        z = a_h_biased @ weights[i + 1]
        a = vectorized_func(z, sigmoid)
        Z.append(z)
        A.append(a)

    return Z, A
