import numpy as np
from net.layers.layer import Layer

class Reshape(Layer):
    def __init__(self, input_shape, output_shape=None):
        if not output_shape:
            output_shape = (1, np.prod(input_shape))
        super().__init__(input_shape, output_shape, trainable=False)

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_error):
        return np.reshape(output_error, self.input_shape), None
