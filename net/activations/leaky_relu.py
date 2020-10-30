import numpy as np
from net.layers.activation import Activation

class LeakyReLU(Activation):
    def __init__(self, p=0.3, **kwargs):
        leaky_relu = lambda x: ((x > 0) * x) + ((x <= 0) * p * x)
        leaky_relu_prime = lambda x: (x > 0) + ((x <= 0) * p)
        super().__init__(leaky_relu, leaky_relu_prime, **kwargs)
