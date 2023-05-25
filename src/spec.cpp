#include <array>
#include <algorithm>
#include <filesystem>
#include <functional>
#include <math.h>
#include <string>
#include <vector>
#include <unordered_map>

#include <Eigen/Dense>
#include <fmt/core.h>
#include <fmt/color.h>
#include <toml++/toml.h>

#include "spec.h"
#include "network.h"

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

    learning_rates = std::vector<double>(0);
    regularisation_rates = std::vector<double>(0);
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

    auto learning_rates_toml = spec_file["network"]["hyperparameters"]["learning_rates"].as_array();
    for (size_t i = 0; i < learning_rates_toml->size(); ++i) {
        learning_rates.push_back(learning_rates_toml->get_as<double>(i)->value_or(0.0));
    }

    auto regularisation_rates_toml = spec_file["network"]["hyperparameters"]["regularisation_rates"].as_array();
    for (size_t i = 0; i < regularisation_rates_toml->size(); ++i) {
        regularisation_rates.push_back(regularisation_rates_toml->get_as<double>(i)->value_or(0.0));
    }

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

    auto learning_rates_toml = spec_file["network"]["hyperparameters"]["learning_rates"].as_array();
    for (size_t i = 0; i < learning_rates_toml->size(); ++i) {
        learning_rates.push_back(learning_rates_toml->get_as<double>(i)->value_or(0.0));
    }

    auto regularisation_rates_toml = spec_file["network"]["hyperparameters"]["regularisation_rates"].as_array();
    for (size_t i = 0; i < regularisation_rates_toml->size(); ++i) {
        regularisation_rates.push_back(regularisation_rates_toml->get_as<double>(i)->value_or(0.0));
    }

    activation_function = spec_file["network"]["activation_f"].value<std::string>().value_or("Linear");
    classification_function = spec_file["network"]["classification_f"].value<std::string>().value_or("Linear");
    std::transform(activation_function.begin(), activation_function.end(), activation_function.begin(), ::toupper);
    std::transform(classification_function.begin(), classification_function.end(), classification_function.begin(), ::toupper);
}

void NeuralNetworkSpecification::create_networks() {
    const size_t num_networks = learning_rates.size() * regularisation_rates.size();

    size_t learning_idx = 0;
    size_t regularisation_idx = 0;
    for (size_t i = 0; i < num_networks; ++i) {
        networks.push_back(
            NeuralNetwork(id, structure, network_functions.at(activation_function), network_functions.at(classification_function), learning_rates[learning_idx], regularisation_rates[regularisation_idx])
        );

        ++learning_idx;
        if (learning_idx == learning_rates.size()) {
            learning_idx = 0;
            ++regularisation_idx;
        }
        if (regularisation_idx == regularisation_rates.size()) {
            regularisation_idx = 0;
        }
    }
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
    fmt::print(fg(fmt::color::gold), "\tLearning Rates "); fmt::println("{}", fmt::join(learning_rates, " "));
    fmt::print(fg(fmt::color::gold), "\tRegularisation Rates "); fmt::println("{}", fmt::join(regularisation_rates, " "));
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