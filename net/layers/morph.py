import numpy as np

class MorphLayer:
    def __init__(self, output_shape=(1, -1)):
        self.input_shape = None
        self.output_shape = output_shape

    def forward(self, input):
        if not self.input_shape:
            self.input_shape = np.shape(input)
        return np.reshape(input, self.output_shape)
    
    def backward(self, output_error):
        return np.reshape(output_error, self.input_shape)
    
    def update(self, learning_rate):
        pass
