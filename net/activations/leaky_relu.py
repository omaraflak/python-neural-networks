import numpy as np
from net.layers.activation import Activation

def leaky_relu(x):
    return ((x > 0) * x) + ((x <= 0) * p * x)

def leaky_relu_prime(x):
    return (x > 0) + ((x <= 0) * p)

class LeakyReLU(Activation):
    def __init__(self, p=0.3, **kwargs):
        super().__init__(leaky_relu, leaky_relu_prime, **kwargs)
