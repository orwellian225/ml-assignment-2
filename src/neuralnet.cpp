#include <random>

#include <fmt/core.h>
#include <Eigen/Dense>
#include <Eigen/Core>

#include "neuralnet.h"

nnlayer_t init_layer(size_t in_count, size_t out_count) {
    Eigen::MatrixXd weights(out_count, in_count + 1);

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> uniform_dist(0.0, 2.0);

    for (size_t i = 0; i < in_count + 1; ++i) {
        for (size_t j = 0; j < out_count; ++j) {
            weights(j, i) = uniform_dist(rng);
        }
    }

    return nnlayer_t {
        in_count, out_count,
        weights
    };
}

Eigen::VectorXd nnlayer_t::fprop_layer(const Eigen::VectorXd& input, std::function<Eigen::VectorXd(Eigen::VectorXd)> activate_f) {
    Eigen::VectorXd input_with_bias(input.size() + 1);
    Eigen::Vector<double, 1> bias(1.0);
    input_with_bias << bias, input; 

    Eigen::VectorXd result = this->weights * input_with_bias;
    result = activate_f(result);

    return result;
}

void nnlayer_t::print() {
    fmt::print(
        "Network Layer:\nInput neuron count {}\nOutput neuron count {}\nWeights\n{}\n",
        this->input_count, this->output_count, this->weights
    );
}

nn_t init_network(std::vector<size_t> network_structure) {
    std::vector<nnlayer_t> layers;

    for (size_t l = 1; l < network_structure.size(); ++l) {
        nnlayer_t new_layer = init_layer(network_structure[l-1], network_structure[l]);
        layers.push_back(new_layer);
    }

    size_t max_layer_node = *std::max_element(network_structure.begin(), network_structure.end());
    size_t layer_count = network_structure.size();

    return nn_t {
        network_structure,
        layers,
        max_layer_node,
        layer_count
    };
}

Eigen::MatrixXd nn_t::fprop_network(const Eigen::VectorXd& input, std::function<Eigen::VectorXd(Eigen::VectorXd)> activate_f) {
    Eigen::MatrixXd activation_matrix(layer_count, max_layer_nodes);

    activation_matrix.row(0) = input;
    for (size_t l = 1; l < layer_count; ++l) {
        activation_matrix.row(l) = layers[l].fprop_layer(activation_matrix.row(l - 1), activate_f);
    }

    return activation_matrix;
}

Eigen::VectorXd nn_t::eval_network(const Eigen::VectorXd& input, std::function<Eigen::VectorXd(Eigen::VectorXd)> activate_f) {
    Eigen::MatrixXd activation_matrix = fprop_network(input, activate_f);
    return activation_matrix.row(layer_count - 1);
}

Eigen::VectorXd nn_t::eval_network(const Eigen::VectorXd& input, std::function<Eigen::VectorXd(Eigen::VectorXd)> activate_f, std::function<Eigen::VectorXd(Eigen::VectorXd)> classify_f) {
    Eigen::VectorXd activation_result = eval_network(input, activate_f);
    return classify_f(activation_result);
}

void nn_t::print() {
    fmt::print("Neural Network:\n");
    fmt::print("Network Structure: ({})\n", fmt::join(structure, " "));
    fmt::print("Network Layers:\n");

    for (auto layer: layers) {
        layer.print();
    }
}