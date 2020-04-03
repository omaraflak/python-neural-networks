import sys
sys.path.append('..')

import numpy as np

from net.layers import FCLayer, SoftmaxLayer, ActivationLayer
from net.activations import tanh, tanh_prime
from net.losses import mse, mse_prime
from net.utils import forward, backward

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]

X = np.reshape(X, (4, 1, 2))
Y = np.reshape(Y, (4, 1, 1))

network = [
    FCLayer(2, 3),
    ActivationLayer(tanh, tanh_prime),
    FCLayer(3, 1),
    ActivationLayer(tanh, tanh_prime)
]

epochs = 1000
learning_rate = 0.1

# training
for epoch in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        # forward
        output = forward(network, x)

        # error (display purpose only)
        error += mse(y, output)

        # backward
        backward(network, mse_prime(y, output))

        # update parameters
        for layer in network:
            layer.update(learning_rate)

    error /= len(X)
    print('%d/%d, error=%f' % (epoch + 1, epochs, error))

# test
print(forward(network, X))
