import numpy as np
from net.layers.layer import Layer

class Softmax(Layer):
    def __init__(self):
        super().__init__(trainable=False)

    def forward(self, input):
        self.input = input
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_error):
        input_error = np.zeros(output_error.shape)
        out = np.tile(self.output.T, output_error.size)
        return self.output * np.dot(output_error, np.identity(output_error.size) - out), None
