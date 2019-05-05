from .back_propagation import back_propagation
from .feed_forward import feed_forward


def gradient_descent(X, Y, h_weights, o_weights, epochs, rate):
    for i in range(0, epochs):
        rows, cols = X.shape
        gradient_h, gradient_o = back_propagation(X, Y, h_weights, o_weights)
        o_weights -= rate * (1 / rows) * gradient_o
        h_weights -= rate * (1 / rows) * gradient_h

        # z_h, z_o, a_h, a_o = feed_forward(X, h_weights, o_weights)
        # print(a_o)
