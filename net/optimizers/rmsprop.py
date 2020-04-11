from .optimizer import OptimizerBase

class RMSprop(OptimizerBase):
    def __init__(self, learning_rate=0.01, decay_rate=0.9, eps=1e-8):
        super().__init__()
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.eps = eps
        self.cache = np.zeros(self.shape)

    def update(self, weights):
        self.cache = self.decay_rate * self.cache + (1 - self.decay_rate) * np.power(weights, 2)
        return -self.learning_rate * weights / (np.sqrt(self.cache) + self.eps)
