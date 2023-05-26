import data_ops as do
import network

data_filepath = "./data/basic_data.txt"
network_filepath = "./data/saved_nn/test.nnw"

def main():
    data = do.read_data(data_filepath)
    nn = network.NeuralNetwork(network_filepath)

    for d in data:
        print(nn.eval(d))


if __name__ == "__main__":
    main()