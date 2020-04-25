import numpy as np
from .initializer import Initializer

class Xavier(Initializer):
    def get(self, *shape):
        io = self.get_io()
        input_neurons = np.prod(io[0])
        output_neurons = np.prod(io[1])
        return np.random.randn(*shape) * np.sqrt(2 / (input_neurons + output_neurons))
