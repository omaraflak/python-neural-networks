import numpy as np
from .initializer import Initializer

class Normal(Initializer):
    def get(self, *shape):
        return np.random.randn(*shape)
