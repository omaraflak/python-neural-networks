import numpy as np
from .optimizer import OptimizerBase

class Momentum(OptimizerBase):
    def __init__(self, learning_rate=0.01, mu=0.95, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.mu = mu
        self.v = np.zeros(self.shape)

    def update(self, iteration, weights):
        self.v = self.mu * self.v + self.learning_rate * weights
        return -self.v
