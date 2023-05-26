import numpy as np
import typing

class NeuralNetwork:
    def __init__(self, num_features: int, num_labels: int, weights: typing.List[np.array], activation_f: typing.Callable[[np.array], np.array], classification_f: typing.Callable[[np.array], np.array]) -> None:
        self.weights = weights
        self.activation_f = activation_f
        self.classification_f = classification_f
        self.num_features = num_features
        self.num_labels = num_labels

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