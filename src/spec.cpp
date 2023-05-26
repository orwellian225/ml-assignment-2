#include <array>
#include <algorithm>
#include <filesystem>
#include <functional>
#include <math.h>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <fmt/core.h>
#include <fmt/color.h>
#include <toml++/toml.h>

#include "spec.h"
#include "network.h"
#include "hyperparams.h"

NeuralNetworkSpecification::NeuralNetworkSpecification() {
    id = "No ID";
    name = "No Name";
    author = "No Author";

    structure = std::vector<size_t>(0);
    data_file = std::filesystem::path("~");
    label_file = std::filesystem::path("~");
    num_features = 0;
    num_labels = 0;
    data_size = 0;

    activation_function = "NONE";
    classification_function = "NONE";

    networks = std::vector<NeuralNetwork>(0);
}

NeuralNetworkSpecification::NeuralNetworkSpecification(std::filesystem::path spec_filepath) {
    toml::table spec_file = toml::parse_file(spec_filepath.string());
    id = "No ID";
    name = spec_file["name"].value<std::string>().value_or("No Name");
    author = spec_file["author"].value<std::string>().value_or("No Author");

    auto structure_arr = spec_file["network"]["structure"].as_array();
    for (size_t i = 0; i < structure_arr->size(); ++i) {
        structure.push_back((size_t)(structure_arr->get_as<int64_t>(i)->value_or(0)));
    }

    data_file = std::filesystem::path(spec_file["data"]["data_file"].value<std::string>().value_or("~"));
    label_file = std::filesystem::path(spec_file["data"]["label_file"].value<std::string>().value_or("~"));
    num_features = spec_file["data"]["feature_count"].value<size_t>().value_or(0);
    num_labels = spec_file["data"]["label_count"].value<size_t>().value_or(0);
    data_size = spec_file["data"]["size"].value<size_t>().value_or(0);
    hyperparam_set = HyperparamSet(*spec_file["network"]["hyperparameters"].as_table());
    activation_function = spec_file["network"]["activation_f"].value<std::string>().value_or("Linear");
    classification_function = spec_file["network"]["classification_f"].value<std::string>().value_or("Linear");
    std::transform(activation_function.begin(), activation_function.end(), activation_function.begin(), ::toupper);
    std::transform(classification_function.begin(), classification_function.end(), classification_function.begin(), ::toupper);
} 

NeuralNetworkSpecification::NeuralNetworkSpecification(toml::table spec_file) {
    id = "No ID";
    name = spec_file["name"].value<std::string>().value_or("No Name");
    author = spec_file["author"].value<std::string>().value_or("No Author");

    auto structure_arr = spec_file["network"]["structure"].as_array();
    for (size_t i = 0; i < structure_arr->size(); ++i) {
        structure.push_back((size_t)(structure_arr->get_as<int64_t>(i)->value_or(0)));
    }

    data_file = std::filesystem::path(spec_file["data"]["data_file"].value<std::string>().value_or("~"));
    label_file = std::filesystem::path(spec_file["data"]["label_file"].value<std::string>().value_or("~"));
    num_features = spec_file["data"]["feature_count"].value<size_t>().value_or(0);
    num_labels = spec_file["data"]["label_count"].value<size_t>().value_or(0);
    data_size = spec_file["data"]["size"].value<size_t>().value_or(0);
    hyperparam_set = HyperparamSet(*spec_file["network"]["hyperparameters"].as_table());
    activation_function = spec_file["network"]["activation_f"].value<std::string>().value_or("Linear");
    classification_function = spec_file["network"]["classification_f"].value<std::string>().value_or("Linear");
    std::transform(activation_function.begin(), activation_function.end(), activation_function.begin(), ::toupper);
    std::transform(classification_function.begin(), classification_function.end(), classification_function.begin(), ::toupper);
}

void NeuralNetworkSpecification::create_networks() {
    const size_t num_networks = hyperparam_set.count_permutations();
    std::vector<hyperparams_t> hp_permutations = hyperparam_set.construct_permutations();

    for (size_t i = 0; i < num_networks; ++i) {
        networks.push_back(NeuralNetwork(id, structure, activation_function, classification_function, hp_permutations[i]));
    }
}

void NeuralNetworkSpecification::train_networks(const Eigen::MatrixXd& data, const Eigen::VectorXd& labels) {
    const size_t num_networks = networks.size();
    const size_t num_data = data.rows();

    const size_t size_training_data = num_data * 0.8;
    const size_t size_validation_data = num_data * 0.1;
    const size_t size_testing_data = num_data * 0.1;

    assert(size_training_data + size_validation_data + size_testing_data == num_data);

    const Eigen::MatrixXd& training_data = data.block(0, 0, size_training_data, num_features); 
    const Eigen::VectorXd& training_labels = labels.block(0, 0, size_training_data, 1); 

    const Eigen::MatrixXd& validation_data = data.block(size_training_data, 0, size_validation_data, num_features);
    const Eigen::VectorXd& validation_labels = labels.block(size_training_data, 0, size_validation_data, 1);

    const Eigen::MatrixXd& testing_data = data.block(size_training_data + size_validation_data, 0, size_testing_data, num_features);
    const Eigen::VectorXd& testing_labels = labels.block(size_training_data + size_validation_data, 0, size_testing_data, 1);

    std::vector<Eigen::MatrixXi> network_confusion_matricies(num_networks);
    std::vector<double> network_accuracies(num_networks);

    fmt::println("Before Network Performance");
    for (size_t i = 0; i < num_networks; ++i) {
        network_confusion_matricies[i] = networks[i].calc_confusion_matrix(validation_data, validation_labels);
        network_accuracies[i] = networks[i].calc_network_accuracy(network_confusion_matricies[i]);
        fmt::println("\t{} | alpha = {}, lambda = {} | {}", i, networks[i].hyperparams.learning_rate, networks[i].hyperparams.regularisation_rate, network_accuracies[i]);
    }
    fmt::println("");

    fmt::println("After Network Performance");
    for (size_t i = 0; i < num_networks; ++i) {
        networks[i].train(training_data, training_labels, hyperparam_set.num_epochs);
        network_confusion_matricies[i] = networks[i].calc_confusion_matrix(validation_data, validation_labels);
        network_accuracies[i] = networks[i].calc_network_accuracy(network_confusion_matricies[i]);
        networks[i].serialize(std::filesystem::path("data\\saved_nn"),fmt::format("nn_{}", i));
        fmt::println("\t{} | alpha = {}, lambda = {} | {}", i, networks[i].hyperparams.learning_rate, networks[i].hyperparams.regularisation_rate, network_accuracies[i]);
    }

    fmt::println("");
}

void NeuralNetworkSpecification::print_info() {
    fmt::print(fg(fmt::color::green), "Network "); fmt::print("{}\n", name);
    fmt::print(fg(fmt::color::green), "================================================================\n");
    fmt::print(fg(fmt::color::orange), "Structure "); fmt::print("{}\n", fmt::join(structure, " "));
    fmt::print(fg(fmt::color::orange), "Data "); fmt::print("{}\n", data_file.string());
    fmt::print(fg(fmt::color::orange), "Labels "); fmt::print("{}\n", label_file.string());
    fmt::print(fg(fmt::color::orange), "Data size "); fmt::println("{}", data_size);
    fmt::print(fg(fmt::color::orange), "Feature count "); fmt::println("{}", num_features);
    fmt::print(fg(fmt::color::orange), "Label count "); fmt::println("{}", num_labels);
    fmt::print(fg(fmt::color::orange), "Activation function "); fmt::println("{}", activation_function);
    fmt::print(fg(fmt::color::orange), "Classification function "); fmt::println("{}", classification_function);
    fmt::print(fg(fmt::color::orange), "Hyperparameters\n");
    fmt::print(fg(fmt::color::gold), "\tLearning Rates "); fmt::println("{}", fmt::join(hyperparam_set.learning_rates, " "));
    fmt::print(fg(fmt::color::gold), "\tRegularisation Rates "); fmt::println("{}", fmt::join(hyperparam_set.regularisation_rates, " "));
    fmt::print(fg(fmt::color::green), "----------------------------------------------------------------\n");
}

void NeuralNetworkSpecification::print_networks() {
    fmt::print(fg(fmt::color::green), "----------------------------------------------------------------\n");
    for (auto network: networks) { 
        fmt::print(fg(fmt::color::green), "{:^64}\n", "--------");
        network.print_info(); 
    }
    fmt::print(fg(fmt::color::green), "----------------------------------------------------------------\n");
}