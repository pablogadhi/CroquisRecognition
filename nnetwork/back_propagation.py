from .feed_forward import feed_forward
from .utils import vectorized_func, sigmoid_d, cross_entropy_loss


def back_propagation(X, Y, h_weights, o_weights, h_bias, o_bias, l_rate):
    a_h, a_o = feed_forward(X, h_weights, o_weights, h_bias, o_bias)

    error_o = (Y - a_o) * vectorized_func(a_o, sigmoid_d)
    error_h = (error_o @ o_weights.T) * vectorized_func(a_h, sigmoid_d)

    h_weights += (X.T @ error_h) * l_rate
    o_weights += (a_h.T @ error_o) * l_rate

    return a_o
