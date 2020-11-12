class Layer:
    def __init__(self, input_shape=None, output_shape=None, trainable=True):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.trainable = trainable
        self.input = None
        self.output = None

    def on_input_shape(self):
        pass

    def initialize(self, initializer):
        pass

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError

    def update(self, updates):
        if self.trainable:
            raise NotImplementedError
