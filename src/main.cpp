#include <math.h>

#include <fmt/core.h>

#include "neuralnet.h"

/* NOTE TO SELF ON CMAKE:
 *  Add new .cpp files to the cmake file and then rebuild the make file
*/

int main() {

    nnlayer_t test_layer = init_layer(5, 2);
    // test_layer.print();

    std::vector<size_t> test_structure{3, 2, 1};
    nn_t test_network = init_network(test_structure);
    test_network.print();

    // auto activation_lu = [](double x) { return x; };
    // auto activation_s = [](double x) { return 1.0 / (1.0 + std::exp(-1.0 * x)); };
    // auto activation_relu = [](double x) { return x > 0 ? x : 0; };

    auto activation_lu = [](const Eigen::VectorXd& values) { return values.unaryExpr([](double x) { return x; }); };
    auto activation_sig = [](const Eigen::VectorXd& values) { return values.unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); }); };
    auto activation_relu = [](const Eigen::VectorXd& values) { return values.unaryExpr([](double x) { return x > 0 ? x : 0; }); };
    auto activation_softmax = [](const Eigen::VectorXd& values) { 
        Eigen::VectorXd exp_vals = values.unaryExpr([](double x) { return std::exp(x); });
        const double exp_sum = exp_vals.sum();
        return exp_vals.unaryExpr([exp_sum](double x) { return x / exp_sum; });
    };

    Eigen::VectorXd input {{2.0, 3.0, 4.0 }};
    // Eigen::VectorXd result_lu = test_layer.fprop_layer(input, activation_lu);
    // Eigen::VectorXd result_sig = test_layer.fprop_layer(input, activation_sig);
    // Eigen::VectorXd result_relu = test_layer.fprop_layer(input, activation_relu);
    // Eigen::VectorXd result_softmax = test_layer.fprop_layer(input, activation_softmax);

    // fmt::print("Input Vector:\n{}\n", input);
    // fmt::print("Linear Unit Result Vector:\n{}\n", result_lu);
    // fmt::print("Sigmoid Unit Result Vector:\n{}\n", result_sig);
    // fmt::print("ReLU Result Vector:\n{}\n", result_relu);
    // fmt::print("Softmax Result Vector:\n{}\n", result_softmax);

    Eigen::VectorXd fprop_result = test_network.eval_network(input, activation_relu);
    fmt::print("Test fprop result\n{}\n", fprop_result);

    return 0;
}