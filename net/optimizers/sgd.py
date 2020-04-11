from .optimizer import OptimizerBase

class SGD(OptimizerBase):
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.learning_rate = learning_rate

    def update(self, iteration, weights):
        return -self.learning_rate * weights
