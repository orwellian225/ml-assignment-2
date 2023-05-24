#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <fmt/ostream.h>

typedef std::function<Eigen::VectorXd(const Eigen::VectorXd&)> NetworkFunc;

template <typename T>
struct fmt::formatter<
  T,
  std::enable_if_t<
    std::is_base_of_v<Eigen::DenseBase<T>, T>, char>> : ostream_formatter {};

class NeuralNetwork {
    public:
        // Metadata
        std::string spec_id;
        std::vector<size_t> structure;

        // Hyperparams
        double learning_rate;
        double regularisation_rate;
        std::vector<size_t> using_features;
        NetworkFunc activate_f;
        NetworkFunc classify_f;

        // Weights
        std::vector<Eigen::MatrixXd> weights;

        NeuralNetwork() : NeuralNetwork("-1", std::vector<size_t>(0), [](const Eigen::VectorXd& values) { return values; }, 0.0) {}
        NeuralNetwork(std::string spec_id, std::vector<size_t> structure, NetworkFunc activate_f, double learning_rate) : NeuralNetwork(spec_id, structure, activate_f, learning_rate, 0.0) {}
        NeuralNetwork(std::string spec_id, std::vector<size_t> structure, NetworkFunc activate_f, double learning_rate, double regularisation_rate) : NeuralNetwork(spec_id, structure, activate_f, learning_rate, regularisation_rate, std::vector<size_t>(0)) {}
        NeuralNetwork(std::string spec_id, std::vector<size_t> structure, NetworkFunc activate_f, double learning_rate, double regularisation_rate, std::vector<size_t> using_features) : NeuralNetwork(spec_id, structure, activate_f, [](const Eigen::VectorXd& values) { return values; }, learning_rate, regularisation_rate, using_features) {}
        NeuralNetwork(std::string spec_id, std::vector<size_t> structure, NetworkFunc activate_f, NetworkFunc classify_f, double learning_rate) : NeuralNetwork(spec_id, structure, activate_f, classify_f, learning_rate, 0.0) {}
        NeuralNetwork(std::string spec_id, std::vector<size_t> structure, NetworkFunc activate_f, NetworkFunc classify_f, double learning_rate, double regularisation_rate) : NeuralNetwork(spec_id, structure, activate_f, classify_f, learning_rate, regularisation_rate, std::vector<size_t>(0)) {}
        NeuralNetwork(std::string spec_id, std::vector<size_t> structure, NetworkFunc activate_f, NetworkFunc classify_f, double learning_rate, double regularisation_rate, std::vector<size_t> using_features);
        ~NeuralNetwork();

        // Evaluation
        Eigen::MatrixXd fprop(const Eigen::VectorXd& input);
        Eigen::MatrixXd bprop(const Eigen::VectorXd& input, size_t correct_result);
        size_t eval(const Eigen::VectorXd& input);

        // Performance
        Eigen::MatrixXi calc_confusion_matrix(const Eigen::MatrixXd& data, const Eigen::VectorXd& labels);
        Eigen::VectorXd calc_label_accuracy(const Eigen::MatrixXi& confusion_matrix);
        double calc_network_accuracy(const Eigen::MatrixXi& confusion_matrix);

        void serialize(const std::filesystem::path filepath);

        void print_all(const Eigen::MatrixXd& data, const Eigen::VectorXd& labels);
        void print_info();
        void print_weights();
        void print_perf(const Eigen::MatrixXd& data, const Eigen::VectorXd& labels);

    private:
        // Dimensions
        size_t layer_count; // How many layers
        size_t max_layer_nodes; // The number of nodes in the largest layer
        size_t num_labels; // Number of output nodes
        size_t num_features; // Number of input node

        Eigen::VectorXd fprop_layer(const Eigen::VectorXd& input, size_t layer);

        Eigen::VectorXd label_int_to_vec(size_t label); // Convert an integer label to a vector label
        size_t label_vec_to_int(const Eigen::VectorXd& vec); // Convert a vector label to integer label
};