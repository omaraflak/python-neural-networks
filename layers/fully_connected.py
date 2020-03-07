import numpy as np

class FCLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size + output_size)
        self.bias = np.random.randn(1, output_size) / np.sqrt(input_size + output_size)
        self.weights_error = []
        self.bias_error = []

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_error):
        input_error = np.dot(output_error, self.weights.T)
        self.weights_error.append(np.dot(self.input.T, output_error))
        self.bias_error.append(output_error)
        return input_error

    def update(self, learning_rate):
        self.weights -= learning_rate * np.sum(self.weights_error, axis=0)
        self.bias -= learning_rate * np.sum(self.bias_error, axis=0)
        self.weights_error = []
        self.bias_error = []
