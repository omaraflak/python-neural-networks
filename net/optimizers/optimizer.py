import numpy as np

class OptimizerBase:
    def __init__(self):
        self.weights = []
        self.shape = None

    def set_weights(self, weights):
        if not self.shape:
            self.shape = np.shape(weights)
        self.weights.append(weights)

    def get_weights(self):
        updated_weights = self.update(np.sum(self.weights, axis=0))
        self.weights = []
        return updated_weights

    def update(self, weights):
        pass

class Optimizer:
    def __init__(self, OptClass, kwargs):
        self.OptClass = OptClass
        self.kwargs = kwargs
        self.optimizers = None

    def set_weights(self, weights):
        if not self.optimizers:
            self.optimizers = [self.OptClass(**self.kwargs) for w in weights]

        for optimizer, w in zip(self.optimizers, weights):
            optimizer.set_weights(w)

    def get_weights(self):
        return [opt.get_weights() for opt in self.optimizers]
