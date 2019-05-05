from mlxtend.data import loadlocal_mnist
import numpy as np
import sys
from nnetwork.gradient_descent import gradient_descent
from nnetwork.utils import normalize_matrix
from nnetwork.feed_forward import feed_forward

np.set_printoptions(threshold=sys.maxsize)

X, Y = loadlocal_mnist(
    images_path='/home/gadhi/Documents/AI/CroquisRecognition/mnist/train-images-idx3-ubyte',
    labels_path='/home/gadhi/Documents/AI/CroquisRecognition/mnist/train-labels-idx1-ubyte')

h_weights = np.random.rand(784, 14)
o_weights = np.random.rand(14, 10)

y_dummy = np.zeros((Y.size, 10))

for i, v in enumerate(Y):
    y_dummy[i, v] = 1

normalized_X = normalize_matrix(X, 0, 255)

gradient_descent(X, y_dummy, h_weights, o_weights, 300, 3)
