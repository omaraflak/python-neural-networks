import numpy as np
from net.layers.layer import Layer

class Recurrent(Layer):
    def __init__(self, output_size, **kwargs):
        super().__init__(output_shape=(1, output_size), **kwargs)
        self.output = np.zeros((1, output_size))
        self.last_output = None

    def initialize(self, initializer):
        input_size, output_size = self.input_shape[1], self.output_shape[1]
        self.weights = initializer.get(input_size, output_size)
        self.weights2 = initializer.get(1, output_size)
        self.bias = initializer.get(1, output_size)

    def forward(self, input):
        self.input = input
        self.last_output = self.output
        self.output = np.dot(input, self.weights) + (self.last_output * self.weights2) + self.bias
        return self.output

    def backward(self, output_error):
        return np.dot(output_error, self.weights.T), [
            np.dot(self.input.T, output_error),
            output_error * self.last_output,
            output_error
        ]

    def update(self, updates):
        self.weights += updates[0]
        self.weights2 += updates[1]
        self.bias += updates[2]
