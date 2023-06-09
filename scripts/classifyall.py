import numpy as np
import data_ops as do
import network

data_filepath = "testdata.txt"
pca_filepath = "pca_matrix.txt"
network_filepath = "c8e0b7c-6.nnw"
output_filepath = "testlabels.txt"

def main():
    data = do.read_data(data_filepath)
    nn = network.NeuralNetwork(network_filepath)
    pca_matrix = do.read_matrix(pca_filepath);

    centred_data = data - np.mean(data, axis=0)
    data = np.matmul(centred_data, pca_matrix)

    output_file = open(output_filepath, 'w')
    for d in data:
        eval_value = nn.eval(d)
        output_file.write(f"{eval_value}\n")

    output_file.close()

if __name__ == "__main__":
    pass
    main()