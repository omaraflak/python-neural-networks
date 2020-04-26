import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

from net.layers import Dense, Reshape
from net.activations import Tanh
from net.losses import MSE
from net.optimizers import SGD
from net.initializers import Xavier
from net.utils import create_model, train, test, forward

def load_data(n):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_train /= 255

    x_test = x_test.astype('float32')
    x_test /= 255

    return x_train[:n], x_test

model = create_model([
    Reshape((1, 784), input_shape=(28, 28)),
    Dense(30),
    Tanh(),
    Dense(16),
    Tanh(),
    Dense(30),
    Tanh(),
    Dense(784),
    Reshape((28, 28))
], Xavier(), SGD, {'learning_rate': 0.1})
mse = MSE()

x_train, x_test = load_data(1000)
train(model, mse, x_train, x_train, epochs=50)
print('error on test set:', test(model, mse, x_test, x_test))

encoder = model[:5]
decoder = model[5:]

f, ax = plt.subplots(5, 3)
for i in range(5):
    code = forward(encoder, x_test[i])
    reconstructed = forward(decoder, code)
    ax[i][0].imshow(x_test[i], cmap='gray')
    ax[i][1].imshow(np.reshape(code, (4, 4)), cmap='gray')
    ax[i][2].imshow(reconstructed, cmap='gray')
plt.show()
