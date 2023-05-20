import data_ops as do
import numpy as np

features, labels = do.read_labelled_data("./data/small_data.txt", "./data/small_labels.txt")

weights = do.read_labelled_data()


# initialises weight matrix for NN
# layer_node_count: no. of nodes in current layer
# next_node_count: no. of nodes in next later
def init_layer(layer_node_count, next_code_count):
    weight_matrix = []
    for i in range(next_code_count):
        row = []
        for j in range(layer_node_count + 1): # additional weight for bias term
            row.append() # get weights from a file(not sure)
        weight_matrix.append(np.array(row))


# initialises NN by creating and initialising weight matrices for each layer
# layer_counts: no. of nodes in each layer of network
def init_network(layer_counts):
    layers = []
    for i in range(len(layer_counts) - 1): # -1: last index -> output layer
        layers.append(init_layer(layer_counts[i], layer_counts[i + 1]))

    return np.array(layers)


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