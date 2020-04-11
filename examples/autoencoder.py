import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

from net.layers import Dense, Activation
from net.activations import tanh, tanh_prime
from net.losses import mse, mse_prime
from net.optimizers import SGD
from net.utils import create_model, train, test, forward

def load_data(n):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
    x_train = x_train.astype('float32')
    x_train /= 255

    x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
    x_test = x_test.astype('float32')
    x_test /= 255

    return x_train[:n], x_test

model = create_model([
    Dense(28 * 28, 30),
    Activation(tanh, tanh_prime),
    Dense(30, 16),
    Activation(tanh, tanh_prime),
    Dense(16, 30),
    Activation(tanh, tanh_prime),
    Dense(30, 28 * 28)
], SGD, {'learning_rate': 0.1})

x_train, x_test = load_data(1000)
train(model, mse, mse_prime, x_train, x_train, epochs=50)
print('error on test set:', test(model, mse, x_test, x_test))

encoder = model[:4]
decoder = model[4:]

f, ax = plt.subplots(5, 3)
for i in range(5):
    image = np.reshape(x_test[i], (28, 28))
    code = forward(encoder, x_test[i])
    reconstructed = np.reshape(forward(decoder, code), (28, 28))
    ax[i][0].imshow(image, cmap='gray')
    ax[i][1].imshow(np.reshape(code, (4, 4)), cmap='gray')
    ax[i][2].imshow(reconstructed, cmap='gray')
plt.show()
