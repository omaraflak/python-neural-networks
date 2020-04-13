import sys
sys.path.append('..')

import numpy as np
from keras.datasets import mnist

from net.layers import Dense
from net.activations import LeakyReLU, Sigmoid
from net.losses import BinaryCrossEntropy
from net.optimizers import SGD
from net.utils import create_model, forward, backward, update

# load mnist dataset
(x_train, y_train), _ = mnist.load_data()
x_train = x_train.astype('float32')
x_train = x_train / 255
x_train = np.reshape([
    x_train[y_train == 0][:500],
    x_train[y_train == 1][:500],
    x_train[y_train == 2][:500],
    x_train[y_train == 3][:500],
    x_train[y_train == 4][:500],
    x_train[y_train == 5][:500],
    x_train[y_train == 6][:500],
    x_train[y_train == 7][:500],
    x_train[y_train == 8][:500],
    x_train[y_train == 9][:500]
], (500 * 10, 28, 28))
np.random.shuffle(x_train)

# generator model
noise_size = 100
G = create_model([
    Dense(noise_size, 200),
    LeakyReLU(0.2),
    Dense(200, 28 * 28),
    Sigmoid()
], Adam, {'learning_rate': 0.0001, 'beta_1': 0.5})

# discriminator
D = create_model([
    Dense(28 * 28, 200),
    LeakyReLU(0.2),
    Dense(200, 1),
    Sigmoid()
], Adam, {'learning_rate': 0.0001, 'beta_1': 0.5})

# params
loss = BinaryCrossEntropy()
epochs = 50
batch_size = 8

# intermediate generation to create GIF
gen_count = 10
seeds = np.random.randn(gen_count, noise_size)
print_fq = 2

# training
G_errors = []
D_errors = []
for epoch in range(epochs):
    G_error, D_error = 0, 0
    for index, x in enumerate(x_train):
        real_image = np.reshape(x, (1, -1))

        # generate image
        noise = np.random.randn(1, noise_size)
        fake_image = forward(G, noise)

        # discriminate real image + backward
        real_predict = forward(D, real_image)
        soft_real = np.random.uniform(0.9, 1)
        backward(D, loss.prime(soft_real, real_predict))

        # discriminate fake image + backward
        fake_predict = forward(D, fake_image)
        dEDdDG = loss.prime(0, fake_predict)
        dEDdG = backward(D, dEDdDG)

        # backward generator
        dDGdG = dEDdG / dEDdDG
        dEGdDG = loss.prime(1, fake_predict)
        backward(G, dDGdG * dEGdDG)

        G_error += loss.call(1, fake_predict)
        D_error += loss.call(soft_real, real_predict) + loss.call(0, fake_predict)

        # update every batch_size times
        if index % batch_size == 0:
            update(G, index + 1)
            update(D, index + 1)

    G_error /= len(x_train)
    D_error /= len(x_train)
    G_errors.append(G_error)
    D_errors.append(D_error)
    print('%d/%d, g_error=%f, d_error=%f' % (epoch + 1, epochs, G_error, D_error))

    if (epoch + 1) % print_fq == 0:
        plt.figure(figsize=(15, 5))
        for i, seed in enumerate(seeds):
            gen = forward(G, seed) * 255
            image = np.reshape(gen, (28, 28))
            plt.subplot(1, gen_count, i + 1)
            plt.imshow(image, cmap='binary')
        plt.savefig('images/epoch_%d.png' % (epoch + 1))
        plt.close()
