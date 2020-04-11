from .optimizer import OptimizerBase

class Adam(OptimizerBase):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.m = np.zeros(self.shape)
        self.v = np.zeros(self.shape)

    def update(self, weights):
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * weights
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.power(weights, 2)
        return -self.learning_rate * self.m / (np.sqrt(self.v) + self.eps)
