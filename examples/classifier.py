import sys
sys.path.append('..')

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from net.layers import Dense, Reshape
from net.activations import Softmax, Tanh
from net.losses import MSE
from net.optimizers import SGD
from net.initializers import Xavier
from net.utils import create_model, train, test, forward

def load_data(n):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = np_utils.to_categorical(y_train)

    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = np_utils.to_categorical(y_test)

    return x_train[:n], y_train[:n], x_test, y_test

model = create_model([
    Reshape((1, 784), input_shape=(28, 28)),
    Dense(50),
    Tanh(),
    Dense(20),
    Tanh(),
    Dense(10),
    Softmax()
], Xavier(), SGD, {'learning_rate': 0.1})
mse = MSE()

x_train, y_train, x_test, y_test = load_data(1000)
train(model, mse, x_train, y_train, epochs=30)
print('error on test set:', test(model, mse, x_test, y_test))
