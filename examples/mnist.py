import sys
sys.path.append('..')

import numpy as np

from keras.datasets import mnist
from keras.utils import np_utils

from net.layers import Dense, Activation
from net.activations import Softmax, tanh, tanh_prime
from net.losses import mse, mse_prime
from net.utils import create_model, forward, backward, update
from net.optimizers import SGD

def load_data(n):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = np_utils.to_categorical(y_train)

    x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = np_utils.to_categorical(y_test)

    return x_train[:n], y_train[:n], x_test, y_test

model = create_model([
    Dense(28 * 28, 50),
    Activation(tanh, tanh_prime),
    Dense(50, 20),
    Activation(tanh, tanh_prime),
    Dense(20, 10),
    Softmax(10)
], SGD, {'learning_rate': 0.1})

epochs = 30
x_train, y_train, x_test, y_test = load_data(1000)

# training
for epoch in range(epochs):
    error = 0
    for x, y_true in zip(x_train, y_train):
        # forward
        output = forward(model, x)

        # error (display purpose only)
        error += mse(y_true, output)

        # backward
        backward(model, mse_prime(y_true, output))

        # update parameters
        update(model)

    error /= len(x_train)
    print('%d/%d, error=%f' % (epoch + 1, epochs, error))

ratio = sum([np.argmax(y) == np.argmax(forward(model, x)) for x, y in zip(x_test, y_test)]) / len(x_test)
error = sum([mse(y, forward(model, x)) for x, y in zip(x_test, y_test)]) / len(x_test)
print('test set TP: %.2f' % ratio)
print('test set MSE: %.4f' % error)
