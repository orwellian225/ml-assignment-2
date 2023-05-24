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

typedef std::function<Eigen::VectorXd(const Eigen::VectorXd&)> NetworkFunc;

static const std::unordered_map<std::string, NetworkFunc> network_functions = {
    {"LINEAR", [](const Eigen::VectorXd& values) { return values; }},
    {"RELU", [](const Eigen::VectorXd& values) { return values.unaryExpr([](double x) { return x > 0 ? x : 0 ; }); }},
    {"LOGISTIC", [](const Eigen::VectorXd& values) { return values.unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); }); }},
    {"SOFTMAX", [](const Eigen::VectorXd& values) { return values.array().exp() / values.array().exp().sum(); }},
};

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

        NeuralNetworkSpecification();
        NeuralNetworkSpecification(std::filesystem::path spec_filepath);
        NeuralNetworkSpecification(toml::table spec_file);

        void print_all();
        void print_info();
        void print_networks();
};