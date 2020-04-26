import numpy as np
from net.layers.layer import Layer

class BatchNormalization(Layer):
    def __init__(self, epsilon=0.001):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = np.random.randn()
        self.beta = np.random.randn()

    def forward(self, input):
        self.input = input
        self.mu = np.mean(input)
        self.sigma2 = np.var(input)
        self.x_hat = (input - self.mu) / np.sqrt(self.sigma2 + self.epsilon)
        return self.gamma * self.x_hat + self.beta

    def backward(self, output_error):
        N = self.input.size
        dx_hat = output_error * self.gamma
        tmp = N * np.sqrt(self.sigma2 + self.epsilon)
        dinput = (N * dx_hat - np.sum(dx_hat, axis=0) - self.x_hat * np.sum(dx_hat * self.x_hat, axis=0)) / tmp
        return dinput, [
            np.sum(output_error * self.x_hat, axis=0),
            np.sum(output_error, axis=0)
        ]

    def update(self, updates):
        self.gamma += updates[0]
        self.beta += updates[1]
