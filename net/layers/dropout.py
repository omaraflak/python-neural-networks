import numpy as np
from net.layers.layer import Layer

class Dropout(Layer):
    def __init__(self, p=0.1):
        super().__init__(trainable=False)
        self.p = p
        self.mask = None

    def forward(self, input):
        self.mask = np.random.rand(*self.input_shape) < self.p
        output = np.copy(input)
        output[self.mask] = 0
        return output

    def backward(self, output_error):
        input_error = np.ones(self.input_shape)
        input_error[self.mask] = 0
        return input_error, None
