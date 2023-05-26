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
        NetworkFunc activate_f;
        NetworkFunc classify_f;
        NetworkFunc activate_fprime;

        // Weights
        std::vector<Eigen::MatrixXd> weights;

        NeuralNetwork() : NeuralNetwork("-1", std::vector<size_t>(0), [](const Eigen::VectorXd& values) { return values; }, [](const Eigen::VectorXd& values) { return Eigen::VectorXd::Constant(values.size(), 1); }, 0.0) {}
        NeuralNetwork(std::string spec_id, std::vector<size_t> structure, NetworkFunc activate_f, NetworkFunc activate_fprime, double learning_rate) : NeuralNetwork(spec_id, structure, activate_f, activate_fprime, learning_rate, 0.0) {}
        NeuralNetwork(std::string spec_id, std::vector<size_t> structure, NetworkFunc activate_f, NetworkFunc activate_fprime, double learning_rate, double regularisation_rate) : NeuralNetwork(spec_id, structure, activate_f, activate_fprime, [](const Eigen::VectorXd& values) { return values; }, learning_rate, regularisation_rate) {}
        NeuralNetwork(std::string spec_id, std::vector<size_t> structure, NetworkFunc activate_f, NetworkFunc activate_fprime, NetworkFunc classify_f, double learning_rate) : NeuralNetwork(spec_id, structure, activate_f, activate_fprime, classify_f, learning_rate, 0.0) {}
        NeuralNetwork(std::string spec_id, std::vector<size_t> structure, NetworkFunc activate_f, NetworkFunc activate_fprime, NetworkFunc classify_f, double learning_rate, double regularisation_rate);
        ~NeuralNetwork();

        // Evaluation
        size_t eval(const Eigen::VectorXd& input);

        // Performance
        Eigen::MatrixXi calc_confusion_matrix(const Eigen::MatrixXd& data, const Eigen::VectorXd& labels);
        Eigen::VectorXd calc_label_accuracy(const Eigen::MatrixXi& confusion_matrix);
        double calc_network_accuracy(const Eigen::MatrixXi& confusion_matrix);

        void train(const Eigen::MatrixXd& data, const Eigen::VectorXd& labels, const size_t num_epochs);

        void serialize(const std::filesystem::path filepath, const std::string name);

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
        std::vector<Eigen::VectorXd> fprop(const Eigen::VectorXd& input);
        std::vector<Eigen::VectorXd> bprop(const std::vector<Eigen::VectorXd>& activations, size_t correct_result);
        std::vector<Eigen::VectorXd> calc_unreg_gradients(const std::vector<Eigen::VectorXd>& activations, const std::vector<Eigen::VectorXd>& errors);

        Eigen::VectorXd label_int_to_vec(size_t label); // Convert an integer label to a vector label
        size_t label_vec_to_int(const Eigen::VectorXd& vec); // Convert a vector label to integer label
};