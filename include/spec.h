#pragma once

#include <array>
#include <filesystem>
#include <functional>
#include <math.h>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <toml++/toml.h>

#include "network.h"
#include "hyperparams.h"

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

        HyperparamSet hyperparam_set;
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