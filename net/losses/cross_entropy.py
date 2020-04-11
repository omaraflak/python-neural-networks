import numpy as np

def cross_entropy(y_true, y_pred):
    return -np.log(y_pred) if y_true == 1 else -np.log(1 - y_pred)

def cross_entropy_prime(y_true, y_pred):
    return -1 / y_pred if y_true == 1 else y_pred / (1 - y_pred)
