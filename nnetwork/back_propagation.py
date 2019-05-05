from .feed_forward import feed_forward
from .utils import vectorized_func, sigmoid_d, cost


def back_propagation(X, Y, weights):
    Z, A = feed_forward(X, weights)

    layers = len(Z)

    # Delete biases
    for index, z, a in zip(range(0, layers), Z, A):
        rows, cols = z.shape
        Z[index] = z[:rows - index - 1, :]
        A[index] = a[:rows - index - 1, :]

    delta = (A[layers - 1] - Y) * vectorized_func(Z[layers - 1], sigmoid_d)
    gradient_o = A[layers - 2].T @ delta

    gradients = [gradient_o]
    A.insert(0, X)

    for i in range(layers - 2, -1, -1):
        delta = (delta @ weights[i + 1].T) * vectorized_func(Z[i], sigmoid_d)
        gradient_h = A[i].T @ delta
        gradients.insert(0, gradient_h)

    print(cost(A[layers], Y))

    return gradients
