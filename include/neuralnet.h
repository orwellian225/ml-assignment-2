#pragma once

#include <string>

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


    template<typename activationFunction>
    Eigen::VectorXd fprop_layer(Eigen::VectorXd input, activationFunction activate_f);

    void print();
};

nnlayer_t init_layer(size_t in_count, size_t out_count);