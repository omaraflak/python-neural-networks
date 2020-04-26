import numpy as np
from net.layers.activation import Activation

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

class Sigmoid(Activation):
    def __init__(self, **kwargs):
        super().__init__(sigmoid, sigmoid_prime, **kwargs)
