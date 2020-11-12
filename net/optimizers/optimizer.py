import numpy as np

class OptimizerBase:
    def __init__(self, **kwargs):
        self.gradients = []
        self.shape = kwargs['shape']

    def set_gradients(self, gradients):
        self.gradients.append(gradients)

    def get_gradients(self, iteration):
        updated_gradients = self.update(iteration, np.sum(self.gradients, axis=0))
        self.gradients = []
        return updated_gradients

    def update(self, iteration, gradients):
        raise NotImplementedError

class Optimizer:
    def __init__(self, OptimizerBaseClass, optimizerArgs, param_shapes):
        self.optimizers = [
            OptimizerBaseClass(**{**optimizerArgs, 'shape': shape})
            for shape in param_shapes
        ]

    def set_gradients(self, gradients):
        for optimizer, grad in zip(self.optimizers, gradients):
            optimizer.set_gradients(grad)

    def get_gradients(self, iteration):
        return [opt.get_gradients(iteration) for opt in self.optimizers]
