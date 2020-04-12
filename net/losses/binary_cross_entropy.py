import numpy as np
from net.losses.loss import Loss
from net.activations import Sigmoid

class BinaryCrossEntropy(Loss):
    def __init__(self, from_logits=False):
        self.from_logits = from_logits
        self.s = Sigmoid()

    def call(self, y_true, y_pred):
        if self.from_logits:
            return y_true * -np.log(self.s.call(y_pred)) + (1 - y_true) * -np.log(1 - self.s.call(y_pred))
        else:
            return y_true * -np.log(y_pred) + (1 - y_true) * -np.log(1 - y_pred)

    def prime(self, y_true, y_pred):
        if self.from_logits:
            return y_true * -self.s.prime(y_pred) / self.s.call(y_pred) + (1 - y_true) * self.s.prime(y_pred) / (1 - self.s.call(y_pred))
        else:
            return y_true * -1 / y_pred + (1 - y_true) / (1 - y_pred)
