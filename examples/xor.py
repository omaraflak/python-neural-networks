import sys
sys.path.append('..')

import numpy as np

from net.layers import Dense, Activation
from net.activations import tanh, tanh_prime
from net.losses import mse, mse_prime
from net.utils import create_model, forward, backward, update
from net.optimizers import SGD

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 1, 2))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

model = create_model([
    Dense(2, 3),
    Activation(tanh, tanh_prime),
    Dense(3, 1),
    Activation(tanh, tanh_prime)
], SGD, {'learning_rate': 0.1})

epochs = 1000

# training
for epoch in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        # forward
        output = forward(model, x)

        # error (display purpose only)
        error += mse(y, output)

        # backward
        backward(model, mse_prime(y, output))

        # update parameters
        update(model)

    error /= len(X)
    print('%d/%d, error=%f' % (epoch + 1, epochs, error))

# test
for x in X:
    print(forward(model, x))
