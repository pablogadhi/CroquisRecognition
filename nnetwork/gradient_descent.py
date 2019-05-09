from .back_propagation import back_propagation
from .feed_forward import feed_forward


def gradient_descent(X, Y, weights, epochs, rate, compute_gradient):
    for i in range(0, epochs):
        gradients = compute_gradient(X, Y, weights)
        for layer_weights, gradient in zip(weights, gradients):
            layer_weights -= rate * gradient
