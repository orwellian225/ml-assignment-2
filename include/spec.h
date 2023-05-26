#pragma once

#include <array>
#include <filesystem>
#include <functional>
#include <math.h>
#include <string>
#include <vector>
#include <unordered_map>

#include <Eigen/Dense>
#include <toml++/toml.h>

#include "network.h"

class NeuralNetworkSpecification {
    public:
        std::string id;
        std::string name;
        std::string author;

        std::vector<size_t> structure;
        std::filesystem::path data_file;
        std::filesystem::path label_file;
        size_t num_features;
        size_t num_labels;
        size_t data_size;

        std::vector<double> learning_rates;
        std::vector<double> regularisation_rates;
        std::string activation_function;
        std::string classification_function;

        std::vector<NeuralNetwork> networks;

        NeuralNetworkSpecification();
        NeuralNetworkSpecification(std::filesystem::path spec_filepath);
        NeuralNetworkSpecification(toml::table spec_file);

        void create_networks();
        void train_networks(const Eigen::MatrixXd& data, const Eigen::VectorXd& labels);

        void print_all();
        void print_info();
        void print_networks();
};