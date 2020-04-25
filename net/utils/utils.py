from net.optimizers import Optimizer

def create_model(network, initializer, OptimizerClass, optimizerArgs={}):
    # set input_shape & output_shape
    n = len(network)
    for i, layer in enumerate(network):
        if not layer.input_shape and not layer.output_shape:
            layer.input_shape = network[i - 1].output_shape
            layer.output_shape = layer.input_shape
        elif not layer.input_shape:
            layer.input_shape = network[i - 1].output_shape

    # initialize layers
    layer_sizes = [(layer.input_shape, layer.output_shape) for layer in network]
    initializer.set_layer_sizes(layer_sizes)
    for index, layer in enumerate(network):
        initializer.set_layer_index(index)
        layer.initialize(initializer)

    # create one optimizer per layer
    return [
        (layer, Optimizer(OptimizerClass, optimizerArgs) if layer.trainable else None)
        for layer in network
    ]

def summary(model):
    for layer, _ in model:
        print(layer.input_shape, '\t', layer.output_shape)

def forward(model, input):
    output = input
    for layer, _ in model:
        output = layer.forward(output)
    return output

def backward(model, output):
    error = output
    for layer, optimizer in reversed(model):
        error, grad = layer.backward(error)
        if layer.trainable:
            optimizer.set_weights(grad)
    return error

def update(model, iteration):
    for layer, optimizer in model:
        if layer.trainable:
            layer.update(optimizer.get_weights(iteration))

def train(model, loss, x_train, y_train, epochs, batch=1):
    train_set_size = len(x_train)
    for epoch in range(epochs):
        error = 0
        for i, (x, y) in enumerate(zip(x_train, y_train)):
            output = forward(model, x)
            error += loss.call(y, output)
            backward(model, loss.prime(y, output))
            if i % batch == 0:
                update(model, epoch + 1)
        error /= train_set_size
        print('%d/%d, error=%f' % (epoch + 1, epochs, error))

def test(model, loss, x_test, y_test):
    error = 0
    for x, y in zip(x_test, y_test):
        output = forward(model, x)
        error += loss.call(y, output)
    error /= len(x_test)
    return error
