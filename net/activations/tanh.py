import numpy as np
from net.layers.activation import Activation

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.power(np.tanh(x), 2)

class Tanh(Activation):
    def __init__(self, **kwargs):
        super().__init__(tanh, tanh_prime, **kwargs)
