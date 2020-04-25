# My Neural Networks

This is a machine learning __'library'__ I made from scratch, for educational purpose only.

# Example

```python
import numpy as np

from net.layers import Dense, Activation
from net.activations import tanh, tanh_prime
from net.losses import MSE
from net.optimizers import SGD
from net.initializers import Xavier
from net.utils import create_model, train, test

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 1, 2))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

model = create_model([
    Dense(2, 3),
    Activation(tanh, tanh_prime),
    Dense(3, 1),
    Activation(tanh, tanh_prime)
], Xavier(), SGD, {'learning_rate': 0.1})

mse = MSE()
train(model, mse, X, Y, epochs=1000)
print('error on test set:', test(model, mse, X, Y))
```
