import numpy as np
import sys
from nnetwork.gradient_descent import gradient_descent
from nnetwork.back_propagation import back_propagation
from nnetwork.feed_forward import feed_forward
from nnetwork.utils import normalize_matrix

np.set_printoptions(threshold=sys.maxsize)

X = np.load('data/train_data.npy', allow_pickle=True)
Y = np.load('data/train_labels.npy', allow_pickle=True)

# h1_weights = np.random.uniform(-5, 5, (784 + 1, 392))
# h2_weights = np.random.uniform(-5, 5, (392 + 1, 98))
# h3_weights = np.random.uniform(-5, 5, (98 + 1, 10))
# weights = np.array([h1_weights, h2_weights, h3_weights])

weights = np.load('data/weights.npy', allow_pickle=True)

y_dummy = np.zeros((Y.size, 10))

for i, v in enumerate(Y):
    y_dummy[i, v] = 1

normalized_X = normalize_matrix(X, 0, 255)

gradient_descent(X, y_dummy, weights, 500, 0.5, back_propagation)
A = feed_forward(X, weights)
np.save('data/weights.npy', weights, allow_pickle=True)
