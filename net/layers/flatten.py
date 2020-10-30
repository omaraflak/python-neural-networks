import numpy as np
from net.layers.layer import Layer

class Flatten(Layer):
    def __init__(self, **kwargs):
        super().__init__(trainable=False, **kwargs)

    def on_input_shape(self):
        self.output_shape = (1, np.prod(self.input_shape))

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient):
        return np.reshape(output_gradient, self.input_shape), None
