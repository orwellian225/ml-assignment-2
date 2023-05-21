#include <random>
#include <string>
#include <vector>
#include <array>

#include <fmt/core.h>
#include <fmt/color.h>
#include <Eigen/Dense>
#include <Eigen/Core>

#include "neuralnet.h"

nnlayer_t init_layer(size_t layer_id, size_t in_count, size_t out_count) {
    Eigen::MatrixXd weights(out_count, in_count + 1);

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> uniform_dist(-1.0, 1.0);

    for (size_t i = 0; i < in_count + 1; ++i) {
        for (size_t j = 0; j < out_count; ++j) {
            weights(j, i) = uniform_dist(rng);
        }
    }

    return nnlayer_t {
        layer_id, in_count, out_count,
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
        nnlayer_t new_layer = init_layer(l-1, network_structure[l-1], network_structure[l]);
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

Eigen::VectorXd label_to_vector(size_t label, size_t num_labels) {
    Eigen::VectorXd result(num_labels);
    result(label) = 1;
    return result;
}

size_t vector_to_label(const Eigen::VectorXd& label_vec) {
    size_t max_i;
    label_vec.maxCoeff(&max_i);
    return max_i;
}

Eigen::MatrixXd nn_t::fprop_network(const Eigen::VectorXd& input, std::function<Eigen::VectorXd(Eigen::VectorXd)> activate_f) {
    Eigen::MatrixXd activation_matrix(max_layer_nodes, layer_count);

    const size_t initial_input_neuron_count = input.size();
    activation_matrix.block(0, 0, initial_input_neuron_count, 1) = input;
    for (size_t l = 1; l < layer_count; ++l) {
        const size_t input_neuron_count = layers[l - 1].input_count;
        const size_t output_neuron_count = layers[l - 1].output_count;

        Eigen::VectorXd fprop_result = layers[l - 1].fprop_layer(activation_matrix.block(0, l - 1, input_neuron_count, 1), activate_f);
        activation_matrix.block(0, l, output_neuron_count, 1) = fprop_result;
    }

    return activation_matrix;
}

Eigen::VectorXd nn_t::eval_network(const Eigen::VectorXd& input, std::function<Eigen::VectorXd(Eigen::VectorXd)> activate_f) {
    Eigen::MatrixXd activation_matrix = fprop_network(input, activate_f);
    const size_t output_neuron_count = layers[layers.size() - 1].output_count;
    return activation_matrix.block(0, layer_count - 1, output_neuron_count, 1);
}

Eigen::VectorXd nn_t::eval_network(const Eigen::VectorXd& input, std::function<Eigen::VectorXd(Eigen::VectorXd)> activate_f, std::function<Eigen::VectorXd(Eigen::VectorXd)> classify_f) {
    Eigen::VectorXd activation_result = eval_network(input, activate_f);
    return classify_f(activation_result);
}

std::string nn_t::eval_network_perf(size_t label_count, const Eigen::MatrixXd& data, const Eigen::VectorXd labels, std::function<Eigen::VectorXd(Eigen::VectorXd)> activate_f) {
    std::string result = fmt::format(fg(fmt::color::teal), "\nNetwork Performance\n----------------------------------------------------------------\n");

    Eigen::MatrixXi confusion_matrix = Eigen::MatrixXi::Constant(label_count, label_count, 0);
    size_t total_count = data.rows();
    size_t correct_count = 0;

    for (size_t i = 0; i < data.rows(); ++i) {
        Eigen::VectorXd row = data.row(i);
        Eigen::VectorXd nn_result = eval_network(row, activate_f);
        size_t result_class = vector_to_label(nn_result);
        confusion_matrix((size_t)labels(i), result_class) += 1;
    }

    for (size_t i = 0; i < label_count; ++i) {
        correct_count += confusion_matrix(i,i);
    }
    double accuracy = (double)(correct_count) / (double)(total_count) * 100.0;

    result += fmt::format(fg(fmt::color::gold), "Confusion Matrix\n");
    result += fmt::format("{}", confusion_matrix);
    result += "\n";

    result += fmt::format(fg(fmt::color::gold), "Accuracy "); 
    result += fmt::format("{} %\n", accuracy);
    result += fmt::format(fg(fmt::color::teal), "----------------------------------------------------------------\n");
    return result;
}

void nn_t::print() {
    fmt::print("Neural Network:\n");
    fmt::print("Network Structure: ({})\n", fmt::join(structure, " "));
    fmt::print("Network Layers:\n");

    for (auto layer: layers) {
        layer.print();
    }
}