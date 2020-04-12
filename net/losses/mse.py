import numpy as np
from net.losses.loss import Loss

class MSE(Loss):
    def call(self, y_pred, y_true):
        return np.mean(np.power(y_true - y_pred, 2))

    def prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_pred.size
