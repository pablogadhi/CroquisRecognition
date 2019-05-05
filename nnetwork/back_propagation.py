from .feed_forward import feed_forward
from .utils import vectorized_func, sigmoid_d, cost


def back_propagation(X, Y, h_weights, o_weights):
    z_h, z_o, a_h, a_o = feed_forward(X, h_weights, o_weights)

    # Delete all biases
    rows, cols = a_o.shape
    a_o = a_o[:rows - 2, :]
    rows, cols = z_o.shape
    z_o = z_o[:rows - 2, :]
    rows, cols = a_h.shape
    a_h = a_h[:rows - 1, :]
    rows, cols = z_h.shape
    z_h = z_h[:rows - 1, :]

    delta_o = (a_o - Y) * vectorized_func(z_o, sigmoid_d)
    delta_h = (delta_o @ o_weights.T) * vectorized_func(z_h, sigmoid_d)

    gradient_o = a_h.T @ delta_o
    gradient_h = X.T @ delta_h

    print(cost(a_o, Y))

    return gradient_h, gradient_o
