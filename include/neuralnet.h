#pragma once

#include <string>
#include <functional>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <fmt/ostream.h>

template <typename T>
struct fmt::formatter<
  T,
  std::enable_if_t<
    std::is_base_of_v<Eigen::DenseBase<T>, T>,
    char>> : ostream_formatter {};

struct nnlayer_t {
    size_t input_count;
    size_t output_count;
    Eigen::MatrixXd weights;

    Eigen::VectorXd fprop_layer(const Eigen::VectorXd& input, std::function<Eigen::VectorXd(Eigen::VectorXd)> activate_f);

    void print();
};

nnlayer_t init_layer(size_t in_count, size_t out_count);

struct nn_t {
  std::vector<size_t> structure;
  std::vector<nnlayer_t> layers;
  size_t max_layer_nodes; // How many nodes are in the largest layer
  size_t layer_count; // How many layers are in the network

  Eigen::MatrixXd fprop_network(const Eigen::VectorXd& input, std::function<Eigen::VectorXd(Eigen::VectorXd)> activate_f);
  Eigen::VectorXd eval_network(const Eigen::VectorXd& input, std::function<Eigen::VectorXd(Eigen::VectorXd)> activate_f);
  
  // classify_f will only be executed on the last layer of the network. Its there to help reduce a non-linearity that doesn't reduce to 1 then be reduced down to 1
  // Mostly going to be used to reduce a ReLU output to a class vector via softmax or sigmoid
  Eigen::VectorXd eval_network(const Eigen::VectorXd& input, std::function<Eigen::VectorXd(Eigen::VectorXd)> activate_f, std::function<Eigen::VectorXd(Eigen::VectorXd)> classify_f);

  void print();
};

nn_t init_network(std::vector<size_t> network_structure);