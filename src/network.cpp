#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <fmt/color.h>
#include <fmt/core.h>

#include "network.h"

NeuralNetwork::NeuralNetwork(std::string spec_id, std::vector<size_t> structure, NetworkFunc activate_f, NetworkFunc classify_f, double learning_rate, double regularisation_rate, std::vector<size_t> using_features) {
    // Assign
    this->spec_id = spec_id;
    this->structure = structure;
    this->activate_f = activate_f;
    this->classify_f = classify_f;
    this->learning_rate = learning_rate;
    this->regularisation_rate = regularisation_rate;
    this->using_features = using_features;

    // Calculate the network dimensions
    this->layer_count = structure.size();
    this->num_features = structure.front();
    this->num_labels = structure.back();
    this->max_layer_nodes = *std::max_element(structure.begin(), structure.end());

    // Init the network weights
    this->weights = {};
    for (size_t l = 0; l < layer_count - 1; ++l) {
        Eigen::MatrixXd layer_weights = Eigen::MatrixXd::Random(structure[l + 1], structure[l] + 1);
        this->weights.push_back(layer_weights);
    }
}
NeuralNetwork::~NeuralNetwork() {}

Eigen::MatrixXd NeuralNetwork::fprop(const Eigen::VectorXd& input) {
    Eigen::MatrixXd activations = Eigen::MatrixXd::Constant(max_layer_nodes, layer_count, 0);

    activations.block(0, 0, num_features, 1) = input;
    for (size_t l = 1; l < layer_count; ++l) {
        const size_t input_neuron_count = weights[l - 1].cols() - 1; // Remove the bias
        const size_t output_neuron_count = weights[l - 1].rows();

        const Eigen::VectorXd& input = activations.block(0, l - 1, input_neuron_count, 1);
        activations.block(0, l, output_neuron_count, 1) = fprop_layer(input, l - 1);
    }

    // Apply the classification function to the last column of the activation matrix
    activations.block(0, layer_count - 1, num_labels, 1) = classify_f(activations.block(0, layer_count - 1, num_labels, 1));

    return activations;
}

Eigen::MatrixXd NeuralNetwork::bprop(const Eigen::VectorXd& input, size_t correct_label) {
    return Eigen::MatrixXd::Constant(1, 1, 0);
}

size_t NeuralNetwork::eval(const Eigen::VectorXd& input) {
    Eigen::MatrixXd activations = fprop(input);
    return label_vec_to_int(activations.block(0, layer_count - 1, num_labels, 1));
}

Eigen::MatrixXi NeuralNetwork::calc_confusion_matrix(const Eigen::MatrixXd& data, const Eigen::VectorXd& labels) {
    Eigen::MatrixXi confusion = Eigen::MatrixXi::Constant(num_labels, num_labels, 0);
    for (size_t i = 0; i < data.rows(); ++i) {
        size_t evaluation = eval(data.row(i));
        ++confusion((size_t)labels[i], evaluation);
    }

    return confusion;
}

Eigen::VectorXd NeuralNetwork::calc_label_accuracy(const Eigen::MatrixXi& confusion_matrix) {
    auto convert_to_percent = [](double x) { return x * 100.0; };
    Eigen::VectorXd num_correct_evaluations = confusion_matrix.diagonal().cast<double>();
    Eigen::VectorXd num_evaluations = confusion_matrix.rowwise().sum().cast<double>();
    return num_correct_evaluations.array() / num_evaluations.array() * 100.0;
}

double NeuralNetwork::calc_network_accuracy(const Eigen::MatrixXi& confusion_matrix) {
    double num_evaluations = (double)confusion_matrix.sum();
    double num_correct_evaluations = (double)confusion_matrix.diagonal().sum();

    return num_correct_evaluations / num_evaluations * 100.0;
}

void NeuralNetwork::train(const Eigen::MatrixXd& data, const Eigen::VectorXd& labels) {
    return;
}

void NeuralNetwork::serialize(const std::filesystem::path filepath, const std::string name) {
    auto filename = filepath/(fmt::format("{}-{}-alpha{}-lambda{}.nnw", spec_id, name, learning_rate, regularisation_rate));
    FILE* nnw_file = fopen(filename.string().c_str(), "w");
    fmt::println(nnw_file, "{}", fmt::join(structure, ","));

    for (auto weight_m: weights) {
        for (size_t i = 0; i < weight_m.rows(); ++i) {
            for (size_t j = 0; j < weight_m.cols(); ++j) {
                fmt::print(nnw_file, "{},", weight_m(i, j));
            }
            fmt::print(nnw_file, "\n");
        }
    }

    fclose(nnw_file);
}

void NeuralNetwork::print_all(const Eigen::MatrixXd& data, const Eigen::VectorXd& labels) {
    print_info();
    print_perf(data, labels);
    print_weights();
}
void NeuralNetwork::print_info() {
    fmt::print(fg(fmt::color::orange), "Spec: "); fmt::print("{}\n", spec_id);
    fmt::print(fg(fmt::color::orange), "Structure: "); fmt::print("{}\n", fmt::join(structure, " "));
    fmt::print(fg(fmt::color::orange), "\tNumber of layers: "); fmt::print("{}\n", layer_count);
    fmt::print(fg(fmt::color::orange), "\tLargest layer node count: "); fmt::print("{}\n", max_layer_nodes);
    fmt::print(fg(fmt::color::orange), "\tNumber of features (Input Nodes): "); fmt::print("{}\n", num_features);
    fmt::print(fg(fmt::color::orange), "\tNumber of labels (Output Nodes): "); fmt::print("{}\n", num_labels);
    fmt::print(fg(fmt::color::orange), "\tNumber of Hidden layers: "); fmt::print("{}\n", structure.size() - 2);
    fmt::print(fg(fmt::color::orange), "Learning Rate: "); fmt::print("{}\n", learning_rate);
    fmt::print(fg(fmt::color::orange), "Regularisation Rate: "); fmt::print("{}\n", regularisation_rate);
}
void NeuralNetwork::print_weights() {
    fmt::print(fg(fmt::color::red), "TODO: Print Weights in readable fashion");
}
void NeuralNetwork::print_perf(const Eigen::MatrixXd& data, const Eigen::VectorXd& labels) {
    Eigen::MatrixXi confusion_matrix = calc_confusion_matrix(data, labels);
    Eigen::VectorXd label_accuracy = calc_label_accuracy(confusion_matrix);
    double accuracy = calc_network_accuracy(confusion_matrix);

    fmt::print(fg(fmt::color::orange), "Confusion Matrix\n"); fmt::print("{}\n", confusion_matrix);
    fmt::print(fg(fmt::color::orange), "\nClass Accuracy\n"); fmt::print("{}\n", label_accuracy);
    fmt::print(fg(fmt::color::orange), "\nOverall Accuracy "); fmt::print("{:.2f} %\n", accuracy);
}

Eigen::VectorXd NeuralNetwork::fprop_layer(const Eigen::VectorXd& input, size_t layer) {
    Eigen::VectorXd input_with_bias(input.size() + 1);
    Eigen::Vector<double, 1> bias(1.0);
    input_with_bias << bias, input;

    return activate_f(weights[layer] * input_with_bias);
}

Eigen::VectorXd NeuralNetwork::label_int_to_vec(size_t label) {
    Eigen::VectorXd result = Eigen::VectorXd::Constant(num_labels, 0);
    result(label) = 1;
    return result;
}

size_t NeuralNetwork::label_vec_to_int(const Eigen::VectorXd& vec) {
    size_t max_i;
    vec.maxCoeff(&max_i);
    return max_i;
}
