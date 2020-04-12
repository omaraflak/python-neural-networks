import numpy as np
from net.losses.loss import Loss

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

class BinaryCrossEntropy(Loss):
    def __init__(self, from_logits=False):
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            return -np.log(sigmoid(y_pred)) if y_true == 1 else -np.log(1 - sigmoid(y_pred))
        else:
            return -np.log(y_pred) if y_true == 1 else -np.log(1 - y_pred)

    def prime(self, y_true, y_pred):
        if self.from_logits:
            return -sigmoid_prime(y_pred) / sigmoid(y_pred) if y_true == 1 else sigmoid_prime(y_pred) / (1 - sigmoid(y_pred))
        else:
            return -1 / y_pred if y_true == 1 else 1 / (1 - y_pred)
