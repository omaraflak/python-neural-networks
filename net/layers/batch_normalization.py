import numpy as np
from net.layers.layer import Layer

class BatchNormalization(Layer):
    def __init__(self, epsilon=0.001, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def initialize(self, initializer):
        self.gamma = initializer.get()
        self.beta = initializer.get()
        return [(1), (1)]

    def forward(self, input):
        self.input = input
        self.mu = np.mean(input)
        self.sigma2 = np.var(input)
        self.x_hat = (input - self.mu) / np.sqrt(self.sigma2 + self.epsilon)
        return self.gamma * self.x_hat + self.beta

    def backward(self, output_gradient):
        N = self.input.size
        dx_hat = output_gradient * self.gamma
        tmp = N * np.sqrt(self.sigma2 + self.epsilon)
        input_gradient = (N * dx_hat - np.sum(dx_hat, axis=0) - self.x_hat * np.sum(dx_hat * self.x_hat, axis=0)) / tmp
        return input_gradient, [
            np.sum(output_gradient * self.x_hat, axis=0),
            np.sum(output_gradient, axis=0)
        ]

    def update(self, updates):
        self.gamma += updates[0]
        self.beta += updates[1]
