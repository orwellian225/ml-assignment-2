import data_ops as do
import numpy as np
import json


features, labels = do.read_labelled_data("./data/small_data.txt", "./data/small_labels.txt")

file = open('data/weights.json')
data = json.load(file)

weights = []
for i in data['weights']:
    weights.append(np.array(i))

print(weights)

# Forward propagation for a given slice of NN
# Returns output vector of the network until the last layer of the provided slice
# network_slice: slice of NN (subset of layers)
def fprop_layer(network_slice, input, activation_f):
    for layer in network_slice:
        input = np.append(input, 1)

        output = np.matmul(layer, input)
        for i in range(len(output)):
            output[i] = activation_f(output[i])
        input = output

    return output


# Forward propagtion of entire NN
# Returns the matrix of activation values for the network
# last layer are the results of the network
# network: NN: list of weight matrices for each layer
def fprop_network(network, input, activation_f):
    result = []
    for layer in network:
        input = np.append(input, 1)

        output = np.matmul(layer, input)
        for i in range(len(output)):
            output[i] = activation_f(output[i])
        result.append(output)
        input = output

    return result

# Returns the result of the neural network evaluation
def eval_network(network, input, activation_f):
    return fprop_network(network, input, activation_f)[-1]
