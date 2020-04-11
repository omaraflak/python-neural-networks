import numpy as np
from net.layers.layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__((1, input_size), (1, output_size))
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size + output_size)
        self.bias = np.random.randn(1, output_size) / np.sqrt(input_size + output_size)

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_error):
        return np.dot(output_error, self.weights.T), [
            np.dot(self.input.T, output_error),
            output_error
        ]

    def update(self, updates):
        self.weights += updates[0]
        self.bias += updates[1]
