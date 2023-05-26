import data_ops as do
import network

data_filepath = ""
network_filepath = ""
output_filepath = ""

def main():
    data = do.read_data(data_filepath)
    nn = network.NeuralNetwork(network_filepath)

    # PREPROCESSING GOES HERE

    output_file = open(output_filepath, 'w')
    for d in data:
        eval_value = nn.eval(d)
        print(eval_value)
        output_file.write(f"{eval_value}\n")

    output_file.close()

if __name__ == "__main__":
    pass
    main()