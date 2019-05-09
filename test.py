import numpy as np
import sys
from nnetwork.utils import confidence_and_prediction, score, normalize_matrix
from nnetwork.feed_forward import feed_forward

np.set_printoptions(threshold=sys.maxsize)

X = np.load('data/test_data.npy', allow_pickle=True)
Y = np.load('data/test_labels.npy', allow_pickle=True)

weights = np.load('data/weights.npy', allow_pickle=True)
normalized_X = normalize_matrix(X, 0, 255)

A = feed_forward(X, weights)

dummy_pred = A[len(A) - 1]

rows, cols = dummy_pred.shape

conf_and_pred = confidence_and_prediction(dummy_pred)
print(score(np.array([pred for conf, pred in conf_and_pred]).reshape(rows, 1), Y))
