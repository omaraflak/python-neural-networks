import numpy as np

def parametric_relu(x, p=0.01):
    return ((x > 0) * x) + ((x <= 0) * p * x)

def parametric_relu_prime(x, p=0.01):
    return (x > 0) + ((x <= 0) * p)
