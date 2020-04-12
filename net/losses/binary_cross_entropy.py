import numpy as np
from net.losses.loss import Loss
from net.losses.activations import Sigmoid

class BinaryCrossEntropy(Loss):
    def __init__(self, from_logits=False):
        self.from_logits = from_logits
        self.s = Sigmoid()

    def call(self, y_true, y_pred):
        if self.from_logits:
            return -np.log(self.s.call(y_pred)) if y_true == 1 else -np.log(1 - self.s.call(y_pred))
        else:
            return -np.log(y_pred) if y_true == 1 else -np.log(1 - y_pred)

    def prime(self, y_true, y_pred):
        if self.from_logits:
            return -self.s.prime(y_pred) / self.s.call(y_pred) if y_true == 1 else self.s.prime(y_pred) / (1 - self.s.call(y_pred))
        else:
            return -1 / y_pred if y_true == 1 else 1 / (1 - y_pred)
