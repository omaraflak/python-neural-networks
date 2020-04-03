import numpy as np

def relu(x):
    return np.maximum(x, 0)

def relu_prime(x):
    return np.array(x > 0).astype('int')
