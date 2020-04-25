import numpy as np
from .initializer import Initializer

class Xavier(Initializer):
    def get(self, *shape):
        io = self.get_io()
        input_neurons = np.prod(io[0])
        return np.random.randn(*shape) * np.sqrt(1 / input_neurons)
