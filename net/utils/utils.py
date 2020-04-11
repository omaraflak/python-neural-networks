from net.optimizers import Optimizer

def create_model(network, OptimizerBaseClass, kwargs):
    # set input_shape & output_shape
    n = len(network)
    for i, layer in enumerate(network):
        if not layer.input_shape and not layer.output_shape:
            layer.input_shape = network[i - 1].output_shape
            layer.output_shape = layer.input_shape
        elif not layer.input_shape:
            layer.input_shape = network[i - 1].output_shape
        elif not layer.output_shape:
            if i < n - 1:
                layer.output_shape = network[i + 1].input_shape
            else:
                layer.output_shape = layer.input_shape

    # create one optimizer per layer
    return network, [
        Optimizer(OptimizerBaseClass, kwargs) if layer.trainable else None
        for layer in network
    ]

def summary(model):
    for layer in model[0]:
        print(layer.input_shape, '\t', layer.output_shape)

def forward(model, input):
    output = input
    for layer in model[0]:
        output = layer.forward(output)
    return output

def backward(model, output):
    network, optimizers = model
    error = output
    for layer, optimizer in zip(reversed(network), reversed(optimizers)):
        error, grad = layer.backward(error)
        if layer.trainable:
            optimizer.set_weights(grad)
    return error

def update(model):
    for layer, optimizer in zip(*model):
        if layer.trainable:
            layer.update(optimizer.get_weights())
