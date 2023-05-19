#include <random>

#include <fmt/core.h>
#include <Eigen/Dense>
#include <Eigen/Core>

#include "neuralnet.h"

nnlayer_t init_layer(size_t in_count, size_t out_count) {
    Eigen::MatrixXd weights(out_count, in_count + 1);

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> uniform_dist(0.0, 2.0);

    for (size_t i = 0; i < in_count + 1; ++i) {
        for (size_t j = 0; j < out_count; ++j) {
            weights(j, i) = uniform_dist(rng);
        }
    }

    return nnlayer_t {
        in_count, out_count,
        weights
    };
}

Eigen::VectorXd nnlayer_t::fprop_layer(Eigen::VectorXd input, std::function<double(double)> activate_f) {
    Eigen::VectorXd input_with_bias(input.size() + 1);
    Eigen::Vector<double, 1> bias(1.0);
    input_with_bias << bias, input; 

    Eigen::VectorXd result = this->weights * input_with_bias;
    result = result.unaryExpr(activate_f);

    return result;
}

void nnlayer_t::print() {
    fmt::print(
        "Network Layer:\nInput neuron count {}\nOutput neuron count {}\nWeights\n{}\n",
        this->input_count, this->output_count, this->weights
    );
}