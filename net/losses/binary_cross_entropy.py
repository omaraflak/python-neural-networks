import numpy as np
from net.losses.loss import Loss

class BinaryCrossEntropy(Loss):
    def call(self, y_pred, y_true):
        return -np.log(y_pred) if y_true == 1 else -np.log(1 - y_pred)

    def prime(self, y_true, y_pred):
        return -1 / y_pred if y_true == 1 else y_pred / (1 - y_pred)
