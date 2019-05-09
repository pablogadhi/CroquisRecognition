from mlxtend.data import loadlocal_mnist
import numpy as np
import sys
from nnetwork.utils import confidence_and_prediction, score, normalize_matrix
from nnetwork.feed_forward import feed_forward

np.set_printoptions(threshold=sys.maxsize)

X, Y = loadlocal_mnist(
    images_path='/home/gadhi/Documents/AI/CroquisRecognition/mnist/t10k-images-idx3-ubyte',
    labels_path='/home/gadhi/Documents/AI/CroquisRecognition/mnist/t10k-labels-idx1-ubyte')

X = X[:1000, :]
Y = Y[:1000]

weights = np.load('weights.npy', allow_pickle=True)
normalized_X = normalize_matrix(X, 0, 255)

A = feed_forward(X, weights)

dummy_pred = A[len(A) - 1]
# dummy_pred = dummy_pred[:len(dummy_pred) - len(A), :]

conf_and_pred = confidence_and_prediction(dummy_pred)
print(score(np.array([pred for conf, pred in conf_and_pred]), Y))
