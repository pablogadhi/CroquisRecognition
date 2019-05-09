from .feed_forward import feed_forward
from .utils import vectorized_func, sigmoid_d, cost


def back_propagation(X, Y, weights):
    A = feed_forward(X, weights)

    layers = len(A)
    rows, cols = X.shape

    delta = (A[layers - 1] - Y)
    gradient_o = (1 / rows) * (A[layers - 2].T @ delta)

    gradients = [gradient_o]

    for i in range(2, layers):
        sgd = vectorized_func(A[-i][:, 1:], sigmoid_d)
        delta = (delta @ weights[-i + 1].T[:, 1:]) * sgd
        gradient_h = (1 / rows) * (A[-i - 1].T @ delta)
        gradients.insert(0, gradient_h)

    print(cost(A[layers - 1], Y))

    return gradients
