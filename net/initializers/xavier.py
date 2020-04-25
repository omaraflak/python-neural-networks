import numpy as np
from .initializer import Initializer

class Xavier(Initializer):
    def __init__(self, **kwargs):
        super().__init__()

    def get(*shape):
        io = self.layer_sizes[self.index]
        input_neurons = np.prod(io[0])
        output_neurons = np.prod(io[1])
        return np.random.randn(*shape) / np.sqrt(input_neurons + output_neurons)
