import numpy as np
from net.layers.layer import Layer

class Reshape(Layer):
    def __init__(self, output_shape, **kwargs):
        super().__init__(output_shape=output_shape, trainable=False, **kwargs)

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient):
        return np.reshape(output_gradient, self.input_shape), None
