from net.layers.layer import Layer

class Activation(Layer):
    def __init__(self, activation, activation_prime, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backward(self, output_gradient):
        return output_gradient * self.activation_prime(self.input), None
