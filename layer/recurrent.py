import numpy as np

class RLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size + output_size)
        self.weights2 = np.random.randn(1, output_size) / np.sqrt(input_size + output_size)
        self.bias = np.random.randn(1, output_size) / np.sqrt(input_size + output_size)
        self.output = np.zeros((1, output_size))
        self.last_output = None
        self.weights_error = []
        self.weights2_error = []
        self.bias_error = []

    def forward(self, input):
        self.input = input
        self.last_output = self.output
        self.output = np.dot(input, self.weights) + (self.last_output * self.weights2) + self.bias
        return self.output

    def backward(self, output_error):
        input_error = np.dot(output_error, self.weights.T)
        self.weights_error.append(np.dot(self.input.T, output_error))
        self.weights2_error.append(output_error * self.last_output)
        self.bias_error.append(output_error)
        return input_error

    def update(self, learning_rate):
        self.weights -= learning_rate * np.sum(self.weights_error, axis=0)
        self.weights2 -= learning_rate * np.sum(self.weights2_error, axis=0)
        self.bias -= learning_rate * np.sum(self.bias_error, axis=0)
        self.weights_error = []
        self.weights2_error = []
        self.bias_error = []
