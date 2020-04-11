import numpy as np
from net.layers.layer import Layer

class Softmax(Layer):
    def __init__(self, input_size):
        super().__init__((1, input_size), (1, input_size), False)
        self.input_size = input_size

    def forward(self, input):
        self.input = input
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_error):
        input_error = np.zeros(output_error.shape)
        out = np.tile(self.output.T, self.input_size)
        return self.output * np.dot(output_error, np.identity(self.input_size) - out), None
