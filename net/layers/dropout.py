import numpy as np
from net.layers.layer import Layer

class Dropout(Layer):
    def __init__(self, p=0.1, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.p = p
        self.mask = None

    def forward(self, input):
        self.mask = np.random.rand(*self.input_shape) < self.p
        output = np.copy(input)
        output[self.mask] = 0
        return output

    def backward(self, output_gradient):
        input_gradient = np.ones(self.input_shape)
        input_gradient[self.mask] = 0
        return input_gradient, None
