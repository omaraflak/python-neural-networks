import numpy as np
from .initializer import Initializer

class Normal(Initializer):
    def __init__(self, mean=0, std=1):
        super().__init__()
        self.mean = mean
        self.std = std

    def get(self, *shape):
        return np.random.normal(self.mean, self.std, shape)
