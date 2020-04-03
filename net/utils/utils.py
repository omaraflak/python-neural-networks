def forward(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def backward(network, output):
    error = output
    for layer in reversed(network):
        error = layer.backward(error)
    return error
