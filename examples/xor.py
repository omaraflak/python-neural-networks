import sys
sys.path.append('..')

import numpy as np

from net.layers import Dense
from net.activations import Sigmoid
from net.losses import MSE
from net.optimizers import Momentum
from net.initializers import Xavier
from net.utils import create_model, train, test

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 1, 2))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

model = create_model([
    Dense(3, input_shape=(1, 2)),
    Sigmoid(),
    Dense(1),
    Sigmoid()
], Xavier(), Momentum, {'learning_rate': 0.1})
mse = MSE()

train(model, mse, X, Y, epochs=1000)
print('error on test set:', test(model, mse, X, Y))
