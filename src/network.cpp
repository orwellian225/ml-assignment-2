#include <algorithm>
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <fmt/color.h>
#include <fmt/core.h>

#include "network.h"
#include "hyperparams.h"

static const std::unordered_map<std::string, NetworkFunc> network_functions = {
    {"LINEAR", [](const Eigen::VectorXd& values) { return values; }},
    {"RELU", [](const Eigen::VectorXd& values) { return values.unaryExpr([](double x) { return x > 0.0 ? x : 0.0 ; }); }},
    {"LOGISTIC", [](const Eigen::VectorXd& values) { return values.unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); }); }},
    {"SOFTMAX", [](const Eigen::VectorXd& values) { return values.array().exp() / values.array().exp().sum(); }},
};

static const std::unordered_map<std::string, NetworkFunc> network_functions_derivative = {
    {"LINEAR", [](const Eigen::VectorXd& values) { return Eigen::VectorXd::Constant(values.size(), 1); }},
    {"RELU", [](const Eigen::VectorXd& values) { return values.unaryExpr([](double x) { return x > 0.0 ? 1.0 : 0.0 ; }); }},
    {"LOGISTIC", [](const Eigen::VectorXd& values) { return values.unaryExpr([](double x) { return x * (1.0 - x); }); }},
    {"SOFTMAX", [](const Eigen::VectorXd& values) { return values.unaryExpr([](double x) { return x * (1.0 - x); }); }},
};

NeuralNetwork::NeuralNetwork(std::string id, std::vector<size_t> structure, std::string activate_f, std::string classify_f, hyperparams_t hyperparams) {
    // Assign
    this->id = id;
    this->structure = structure;
    this->activate_f = network_functions.at(activate_f);
    this->classify_f = network_functions.at(classify_f);
    this->activate_fprime = network_functions_derivative.at(activate_f);

    this->hyperparams = hyperparams;
    this->activate_f_key = activate_f;
    this->classify_f_key = classify_f;

    // Calculate the network dimensions
    this->layer_count = structure.size();
    this->num_features = structure.front();
    this->num_labels = structure.back();
    this->max_layer_nodes = *std::max_element(structure.begin(), structure.end());

    // Init the network weights
    this->weights = {};
    for (size_t l = 0; l < layer_count - 1; ++l) {
        Eigen::MatrixXd layer_weights = Eigen::MatrixXd::Random(structure[l + 1], structure[l] + 1) / 2;
        this->weights.push_back(layer_weights);
    }
}
NeuralNetwork::~NeuralNetwork() {}


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

size_t NeuralNetwork::eval(const Eigen::VectorXd& input) {
    std::vector<Eigen::VectorXd> activations = fprop(input);
    return label_vec_to_int(activations.back());
}

void NeuralNetwork::train(const Eigen::MatrixXd& data, const Eigen::VectorXd& labels, const size_t num_epochs) {
    std::vector<Eigen::MatrixXd> gradients(layer_count - 1);
    for (size_t i = 0; i < layer_count - 1; ++i) {
        gradients[i] = Eigen::MatrixXd::Constant(weights[i].rows(), weights[i].cols(), 0);
    }

    // The data size needs to be divisable by the batch size 
    assert(data.rows() % hyperparams.batch_size == 0);

    for (size_t e = 0; e < num_epochs; ++e) {    
        bool converged = false;
        for (size_t d = 0; d < data.rows(); d += hyperparams.batch_size) {
            #pragma omp parallel for firstprivate(d)
            for (size_t b = 0; b < hyperparams.batch_size; b++) {
                const size_t data_index = d + b;
                const Eigen::VectorXd& point = data.row(data_index);
                std::vector<Eigen::VectorXd> activations = fprop(point);
                std::vector<Eigen::VectorXd> errors = bprop(activations, labels(data_index));

                // Determine gradients
                const Eigen::Vector<double, 1> bias(1.0);
                for (size_t ii = 0; ii < layer_count - 1; ++ii) {
                    Eigen::VectorXd activation_with_bias(activations[ii].size() + 1);
                    activation_with_bias << bias, activations[ii];
                    gradients[ii] += errors[ii + 1] * activation_with_bias.transpose();
                }
            }

            // Update the weight gradients
            converged = true;
            for (size_t ii = 0; ii < layer_count - 1; ++ii) {
                const Eigen::MatrixXd& prev_weights = gradients[ii];

                // Regularise everything except bias weights
                gradients[ii].block(1, 0, gradients[ii].rows() - 1, gradients[ii].cols()) = gradients[ii].block(1, 0, gradients[ii].rows() - 1, gradients[ii].cols()) / hyperparams.batch_size + hyperparams.regularisation_rate * weights[ii].block(1, 0, weights[ii].rows() - 1, weights[ii].cols());

                // Normalise the bias
                gradients[ii].row(0) /= hyperparams.batch_size;
                
                // Applying update
                weights[ii] -= hyperparams.learning_rate * gradients[ii];

                // Reset gradients
                gradients[ii] = Eigen::MatrixXd::Constant(weights[ii].rows(), weights[ii].cols(), 0);

                converged = converged && (weights[ii] - prev_weights).norm() < hyperparams.convergence_criteria;
            }

            if (converged) {
                break;
            }
        }

        if (converged) {
            break;
        }
    }
}

void NeuralNetwork::serialize(const std::filesystem::path filepath) {
    auto filename = filepath/(fmt::format("{}.nnw", id));
    FILE* nnw_file = fopen(filename.string().c_str(), "w");
    fmt::println(nnw_file, "{},{},{},{}", num_features, num_labels, activate_f_key, classify_f_key);
    fmt::println(nnw_file, "{}", fmt::join(structure, ","));

    for (auto weight_m: weights) {
        for (size_t i = 0; i < weight_m.rows(); ++i) {
            fmt::println(nnw_file, "{}", fmt::join(weight_m.row(i).array(), ","));
        }
    }

    fclose(nnw_file);
}

std::string NeuralNetwork::to_string() {
    std::string result = "";

    result = fmt::format("{:<9} | {:<} | {:<}", fmt::format(fg(fmt::terminal_color::blue), "{}", id), fmt::join(structure, ", "), hyperparams.to_string());

    return result;
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

std::vector<Eigen::VectorXd> NeuralNetwork::fprop(const Eigen::VectorXd& input) {
    std::vector<Eigen::VectorXd> activations(layer_count);

    activations[0] = input;
    for (size_t l = 1; l < layer_count; ++l) {
        const size_t input_neuron_count = weights[l - 1].cols() - 1; // Remove the bias
        const size_t output_neuron_count = weights[l - 1].rows();

        const Eigen::VectorXd& input = activations[l - 1];
        activations[l] = fprop_layer(input, l - 1);
    }

    // Apply the classification function to the last column of the activation matrix
    activations.back() = classify_f(activations.back());

    return activations;
}

std::vector<Eigen::VectorXd> NeuralNetwork::bprop(const std::vector<Eigen::VectorXd>& activations, size_t correct_label) {
    std::vector<Eigen::VectorXd> errors(layer_count);
    Eigen::VectorXd label_vec = label_int_to_vec(correct_label);
    errors.back() = activations.back().array() - label_vec.array();

    for (size_t i = errors.size(); i --> 1;) {
        const Eigen::MatrixXd& unbiased_weights = weights[i - 1].block(0, 1, weights[i - 1].rows(), weights[i - 1].cols() - 1); 
        Eigen::MatrixXd product = unbiased_weights.transpose() * errors[i];
        errors[i - 1] = product.array() * activate_fprime(activations[i - 1]).array();
    }

    return errors;
}