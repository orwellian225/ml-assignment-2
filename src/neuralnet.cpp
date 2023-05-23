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

// Row is the correct label, column is the guessed label
Eigen::MatrixXi nn_t::calc_confusion_matrix(size_t label_count, const Eigen::MatrixXd& data, const Eigen::VectorXd& labels, std::function<Eigen::VectorXd(Eigen::VectorXd)> activate_f) {
    Eigen::MatrixXi confusion_matrix = Eigen::MatrixXi::Constant(label_count, label_count, 0);
    for (size_t i = 0; i < data.rows(); ++i) {
        const Eigen::VectorXd& row = data.row(i);
        Eigen::VectorXd nn_evaluation = eval_network(row, activate_f);
        size_t evaluation_label = vector_to_label(nn_evaluation);
        confusion_matrix((size_t)labels(i), evaluation_label) += 1;
    }

    return confusion_matrix;
}

Eigen::VectorXd nn_t::calc_label_accuracy(const Eigen::MatrixXi& confusion_matrix) {
    Eigen::VectorXd label_accuracies = Eigen::VectorXd::Constant(confusion_matrix.rows(), 0.0);
    for (size_t i = 0; i < confusion_matrix.rows(); ++i) {
        int32_t num_label_evalutions = confusion_matrix.row(i).sum();
        int32_t correct_label_evaluations = confusion_matrix.row(i)(i);
        label_accuracies(i) = (double) correct_label_evaluations / (double) num_label_evalutions * 100.0;
    }

    return label_accuracies;
}

double nn_t::calc_accuracy(const Eigen::MatrixXi& confusion_matrix) {
    int32_t num_evaluations = confusion_matrix.sum();
    int32_t correct_evaluations = 0;
    for (size_t i = 0; i < confusion_matrix.rows(); ++i) {
        correct_evaluations += confusion_matrix(i, i);
    }

    return (double)correct_evaluations / (double)num_evaluations * 100.0;
}

void nn_t::print_perf(size_t label_count, const Eigen::MatrixXd& data, const Eigen::VectorXd labels, std::function<Eigen::VectorXd(Eigen::VectorXd)> activate_f) {
    fmt::print(fg(fmt::color::green), "\nNetwork Performance\n================================================================\n");

    Eigen::MatrixXi confusion_matrix = calc_confusion_matrix(label_count, data, labels, activate_f);
    Eigen::VectorXd label_accuracy = calc_label_accuracy(confusion_matrix);
    double accuracy = calc_accuracy(confusion_matrix);

    fmt::print(fg(fmt::color::orange), "Confusion Matrix\n"); fmt::print("{}\n", confusion_matrix);
    fmt::print(fg(fmt::color::orange), "\nClass Accuracy\n"); fmt::print("{}\n", label_accuracy);
    fmt::print(fg(fmt::color::orange), "\nOverall Accuracy "); fmt::print("{:.2f} %\n", accuracy);
    fmt::print(fg(fmt::color::green), "================================================================\n");
}

void nn_t::print_description() {
    fmt::print("Neural Network:\n");
    fmt::print("Network Structure: ({})\n", fmt::join(structure, " "));
    fmt::print("Network Layers:\n");

    for (auto layer: layers) {
        layer.print();
    }
}