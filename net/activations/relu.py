import numpy as np
from net.layers.activation import Activation

def relu(x):
    return np.maximum(x, 0)

def relu_prime(x):
    return np.array(x > 0).astype('int')

class ReLU(Activation):
    def __init__(self, **kwargs):
        super().__init__(relu, relu_prime, **kwargs)
