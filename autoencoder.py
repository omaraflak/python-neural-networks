import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.utils import np_utils

from layers import FCLayer, SoftmaxLayer, ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime

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

def predict(net, input):
    output = input
    for layer in net:
        output = layer.forward(output)
    return output

network = [
    FCLayer(28 * 28, 30),
    ActivationLayer(tanh, tanh_prime),
    FCLayer(30, 16),
    ActivationLayer(tanh, tanh_prime),
    FCLayer(16, 30),
    ActivationLayer(tanh, tanh_prime),
    FCLayer(30, 28 * 28)
]

epochs = 50
learning_rate = 0.1
x_train, y_train, x_test, y_test = load_data(1000)

# training
for epoch in range(epochs):
    error = 0
    for x in x_train:
        # forward
        output = x
        for layer in network:
            output = layer.forward(output)

        # error (display purpose only)
        error += mse(x, output)

        # backward
        output_error = mse_prime(x, output)
        for layer in reversed(network):
            output_error = layer.backward(output_error)

        # update parameters
        for layer in network:
            layer.update(learning_rate)

    error /= len(x_train)
    print('%d/%d, error=%f' % (epoch + 1, epochs, error))

error = sum([mse(x, predict(network, x)) for x in x_test]) / len(x_test)
print('mse: %.4f' % error)

f, ax = plt.subplots(5, 2)
for i in range(5):
    image = np.reshape(x_train[i], (28, 28))
    reconstructed = np.reshape(predict(network, x_train[i]), (28, 28))
    ax[i][0].imshow(image, cmap='gray')
    ax[i][1].imshow(reconstructed, cmap='gray')
plt.show()
