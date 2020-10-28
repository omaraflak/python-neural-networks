class Initializer:
    def __init__(self):
        self.layer_shapes = None
        self.index = None

    def set_layer_shapes(self, layer_shapes):
        self.layer_shapes = layer_shapes

    def set_layer_index(self, index):
        self.index = index

    def get_io_shape(self):
        return self.layer_shapes[self.index]

    def get(self):
        return self.get(1)[0]

    def get(self, *shape):
        raise NotImplementedError
