import numpy as np
from net.losses.loss import Loss

class SSE(Loss):
    def call(self, y_pred, y_true):
        return 0.5 * np.sum(np.power(y_true - y_pred, 2))

    def prime(self, y_true, y_pred):
        return y_pred - y_true
