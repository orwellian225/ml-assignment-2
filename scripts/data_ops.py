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

"""
Read in labelled data, and then join the feature matrix and label vector into one matrix

Output data format
    AUGMENTED DATA MATRIX
        A joined matrix that has the label for a row as the last column of the row
"""
def read_data_augmented(data_filepath: str, labels_filepath: str) -> np.array:
    raise NotImplementedError