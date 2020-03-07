class ActivationLayer:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backward(self, output_error):
        return output_error * self.activation_prime(self.input)

    def update(self, learning_rate):
        pass
