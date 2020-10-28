import numpy as np
from .optimizer import OptimizerBase

class Adam(OptimizerBase):
    def __init__(self, learning_rate=0.0001, beta_1=0.9, beta_2=0.999, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.m = np.zeros(self.shape)
        self.v = np.zeros(self.shape)

    def update(self, iteration, weights):
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * weights
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.power(weights, 2)
        m_hat = self.m / (1 - np.power(self.beta_1, iteration))
        v_hat = self.v / (1 - np.power(self.beta_2, iteration))
        return -self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
