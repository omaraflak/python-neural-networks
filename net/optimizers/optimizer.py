import numpy as np

class OptimizerBase:
    def __init__(self, **kwargs):
        self.weights = []
        self.shape = kwargs['shape']

    def set_weights(self, weights):
        self.weights.append(weights)

    def get_weights(self, iteration):
        updated_weights = self.update(iteration, np.sum(self.weights, axis=0))
        self.weights = []
        return updated_weights

    def update(self, iteration, weights):
        raise NotImplementedError

class Optimizer:
    def __init__(self, OptimizerBaseClass, optimizerArgs, param_shapes):
        self.optimizers = [
            OptimizerBaseClass(**{**optimizerArgs, 'shape': shape})
            for shape in param_shapes
        ]

    def set_weights(self, weights):
        for optimizer, w in zip(self.optimizers, weights):
            optimizer.set_weights(w)

    def get_weights(self, iteration):
        return [opt.get_weights(iteration) for opt in self.optimizers]
