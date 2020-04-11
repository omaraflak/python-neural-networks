import sys
sys.path.append('..')

import numpy as np

from net.layers import Dense, Activation
from net.activations import tanh, tanh_prime
from net.losses import mse, mse_prime
from net.optimizers import SGD
from net.utils import create_model, train, test

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 1, 2))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

model = create_model([
    Dense(2, 3),
    Activation(tanh, tanh_prime),
    Dense(3, 1),
    Activation(tanh, tanh_prime)
], SGD, {'learning_rate': 0.1})

train(model, mse, mse_prime, X, Y, epochs=1000)
print('error on test set:', test(model, mse, X, Y))
