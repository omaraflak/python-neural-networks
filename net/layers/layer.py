class Layer:
    def __init__(self, input_shape=None, output_shape=None, trainable=True):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.trainable = trainable
        self.input = None
        self.output = None

    def initialize(self, initializer):
        pass

    def forward(self, input):
        pass

    def backward(self, output_error):
        pass

    def update(self, updates):
        pass
