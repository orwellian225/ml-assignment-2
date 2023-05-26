import numpy as np
import typing

network_functions = {
    "LINEAR": lambda np_vec: np_vec,
    "RELU": lambda np_vec: np.vectorize(lambda x: x if x > 0 else 0)(np_vec),
    "LOGISTIC": lambda np_vec: np.vectorize(lambda x: 1 / (1 + np.exp(-x)))(np_vec),
    "SOFTMAX": lambda np_vec: np.exp(np_vec) / np.exp(np_vec).sum()
}

class NeuralNetwork:
    def __init__(self, nn_filepath: str) -> None:
        """ FILE FORMAT
        num_features,num_labels,activation_function,classification_function
        structure as 0,0,0,0
        weights
        """

        nn_file = open(nn_filepath, 'r')

        meta_line = nn_file.readline().strip().split(',')
        num_features = int(meta_line[0])
        num_labels = int(meta_line[1])
        activate_f = meta_line[2]
        classify_f = meta_line[3]

        structure_line = nn_file.readline().strip().split(',')
        structure = np.array(structure_line).astype(np.int32)

        weights = []
        for i in range(len(structure) - 1):
            wmatrix = []
            for j in range(int(structure[i + 1])):
                weights_line = nn_file.readline().strip().split(',')
                wmatrix.append(weights_line)

            weights.append(np.array(wmatrix).astype(np.float64))

        self.weights = weights
        self.structure = structure
        self.activation_f = network_functions[activate_f] 
        self.classification_f = network_functions[classify_f]
        self.num_features = num_features
        self.num_labels = num_labels

        nn_file.close()

    def label_int_to_vec(self, label: int) -> np.array:
        vec = np.array([0] * self.num_labels)
        vec[label] = 1
        return vec

    def label_vec_to_int(self, label: np.array) -> int:
        return label.argmax();

    def fprop_layer(self, input_vec: np.array, layer: int) -> np.array:
        input_with_bias = np.insert(input_vec, 0, 1)
        return self.activation_f(np.matmul(self.weights[layer], input_with_bias))

    def fprop(self, input_vec: np.array) -> typing.List[np.array]:
        activations = []
        activations.append(input_vec)
        for l in range(1, len(self.weights) + 1):
            activations.append(self.fprop_layer(input_vec, l - 1))

        activations[-1] = self.classification_f(activations[-1])
        return activations

    def eval(self, input_vec: np.array) -> np.array:
        activations = self.fprop(input_vec)
        return self.label_vec_to_int(activations[-1])