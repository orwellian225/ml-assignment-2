#include <array>
#include <algorithm>
#include <filesystem>
#include <functional>
#include <math.h>
#include <string>
#include <vector>
#include <chrono>

#include <Eigen/Dense>
#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/chrono.h>
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

NeuralNetworkSpecification::NeuralNetworkSpecification(toml::table spec_file) {
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

    std::string id_prehash = name + author + std::to_string(num_features) + std::to_string(num_labels) + fmt::format("{}", fmt::join(structure, "")) + activation_function + classification_function;
    id = fmt::format("{:x}", std::hash<std::string>{}(id_prehash));
}

void NeuralNetworkSpecification::create_networks() {
    const size_t num_networks = hyperparam_set.count_permutations();
    std::vector<hyperparams_t> hp_permutations = hyperparam_set.construct_permutations();

    for (size_t i = 0; i < num_networks; ++i) {
        networks.push_back(NeuralNetwork(fmt::format("{}-{}", id.substr(0,7), i), structure, activation_function, classification_function, hp_permutations[i]));
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

    fmt::print(fg(fmt::terminal_color::yellow), "Networks\n");
    for (auto network: networks) {
        fmt::println("\t{}", network.to_string());
    }
    fmt::println("");

    fmt::print(fg(fmt::terminal_color::yellow), "Before training network performance\n");
    for (size_t i = 0; i < num_networks; ++i) {
        network_confusion_matricies[i] = networks[i].calc_confusion_matrix(validation_data, validation_labels);
        network_accuracies[i] = networks[i].calc_network_accuracy(network_confusion_matricies[i]);
        fmt::println("\t{} | {} ", fmt::format(fg(fmt::terminal_color::blue), "{}", networks[i].id), network_accuracies[i]);
    }
    fmt::println("");

    auto start_time = std::chrono::system_clock::now();
    fmt::println("{}: {}\n", 
        fmt::format(fg(fmt::terminal_color::yellow), "Started training"), 
        fmt::format(fg(fmt::terminal_color::cyan), "{:%Y-%m-%d %H:%M}", start_time)
    );

    fmt::print(fg(fmt::terminal_color::yellow), "After training network performance\n");
    for (size_t i = 0; i < num_networks; ++i) {
        networks[i].train(training_data, training_labels, hyperparam_set.num_epochs);
        network_confusion_matricies[i] = networks[i].calc_confusion_matrix(validation_data, validation_labels);
        network_accuracies[i] = networks[i].calc_network_accuracy(network_confusion_matricies[i]);
        networks[i].serialize(std::filesystem::path("data\\saved_nn"));
        fmt::println("\t{} | {} ", fmt::format(fg(fmt::terminal_color::blue), "{}", networks[i].id), network_accuracies[i]);
    }
    auto end_time = std::chrono::system_clock::now();
    fmt::println("\n{}: {}", 
        fmt::format(fg(fmt::terminal_color::yellow), "Finished training"), 
        fmt::format(fg(fmt::terminal_color::cyan), "{:%Y-%m-%d %H:%M}", start_time)
    );

    std::chrono::duration<double> elapsed_time = end_time - start_time;
    fmt::println("{}: {}",
        fmt::format(fg(fmt::terminal_color::yellow), "Training took"), 
        fmt::format(fg(fmt::terminal_color::cyan), "{}", elapsed_time)
    );

    fmt::println("");
}