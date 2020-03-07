import numpy as np

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
    FCLayer(28 * 28, 50),
    ActivationLayer(tanh, tanh_prime),
    FCLayer(50, 20),
    ActivationLayer(tanh, tanh_prime),
    FCLayer(20, 10),
    SoftmaxLayer(10)
]

epochs = 30
learning_rate = 0.1
x_train, y_train, x_test, y_test = load_data(1000)

# training
for epoch in range(epochs):
    error = 0
    for x, y_true in zip(x_train, y_train):
        # forward
        output = x
        for layer in network:
            output = layer.forward(output)

        # error (display purpose only)
        error += mse(y_true, output)

        # backward
        output_error = mse_prime(y_true, output)
        for layer in reversed(network):
            output_error = layer.backward(output_error)

        # update parameters
        for layer in network:
            layer.update(learning_rate)

    error /= len(x_train)
    print('%d/%d, error=%f' % (epoch + 1, epochs, error))

ratio = sum([np.argmax(y) == np.argmax(predict(network, x)) for x, y in zip(x_test, y_test)]) / len(x_test)
error = sum([mse(y, predict(network, x)) for x, y in zip(x_test, y_test)]) / len(x_test)
print('ratio: %.2f' % ratio)
print('mse: %.4f' % error)
