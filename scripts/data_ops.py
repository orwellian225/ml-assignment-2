from typing import Tuple
import numpy as np

"""
Read in from the specified files the vector of features as well as the features corresponding label

Input data format:
    DATA FILE
        Comma Seperated
        64 floats
        7 ints

        Each value is a feature
    
    LABEL FILE
        1 int

        The value is the class of the corresponding data found in the DATA FILE

Output data format:
    FEATURE MATRIX
        Numpy matrix

    LABEL VECTOR
        Numpy vector
"""
def read_labelled_data(data_filepath: str, labels_filepath: str, show_log=False) -> Tuple[np.array, np.array]:
    data_file = open(data_filepath, 'r')
    labels_file = open(labels_filepath, 'r')
    
    feature_matrix = [] # Using normal array to construct the matrix and the convert
    label_vector = []

    # Count the number of elements in the data file
    data_count = sum(1 for _ in data_file)
    data_file.seek(0)

    for i in range(data_count):
        if show_log:
            print(f"\rReading data point {i + 1} / {data_count}", end = '')

        data_line = data_file.readline().strip()
        label_line = labels_file.readline().strip()

        feature_vector = np.array(data_line.split(',')).astype(np.float64)

        feature_matrix.append(feature_vector)
        label_vector.append(int(label_line))

    print(f"\r{data_count} / {data_count} data points read.", " "*10, end='\n')    

    data_file.close()
    labels_file.close()

    return np.array(feature_matrix).astype(np.float64), np.array(label_vector).astype(np.uint8)

def read_data(data_filepath: str, show_log=False) -> np.array:
    data_file = open(data_filepath, 'r')

    feature_matrix = []

    # Count the number of elements in the data file
    data_count = sum(1 for _ in data_file)
    data_file.seek(0)

    for i in range(data_count):
        if show_log:
            print(f"\rReading data point {i + 1} / {data_count}", end = '')

        data_line = data_file.readline().strip()

        feature_vector = np.array(data_line.split(',')).astype(np.float64)

        feature_matrix.append(feature_vector)

    print(f"\r{data_count} / {data_count} data points read.", " "*10, end='\n')    

    data_file.close()
    return feature_matrix

def read_network_weights(weights_filepath: str) -> np.array:

    weights_file = open(weights_filepath, 'r')
    structure = weights_file.readline().strip().split(',')
    weights = []

    for i in range(len(structure) - 1):
        wmatrix = []
        for j in range(int(structure[i + 1])):
           wmatrix.append(weights_file.readline().strip().split(','))

        weights.append(np.array(wmatrix))
    
    return weights 
