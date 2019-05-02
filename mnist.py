from mlxtend.data import loadlocal_mnist
import numpy as np
from nnetwork.feed_forward import feed_forward
from nnetwork.back_propagation import back_propagation

X, Y = loadlocal_mnist(
    images_path='/home/gadhi/Documents/AI/CroquisRecognition/mnist/train-images-idx3-ubyte',
    labels_path='/home/gadhi/Documents/AI/CroquisRecognition/mnist/train-labels-idx1-ubyte')

h_weights = np.random.rand(784, 14)
o_weights = np.random.rand(14, 10)

y_dummy = np.zeros((Y.size, 10))

for i, v in enumerate(Y):
    y_dummy[i, v] = 1

# for i in range(0, 30):
pred = back_propagation(X, y_dummy, h_weights, o_weights)
print(pred)