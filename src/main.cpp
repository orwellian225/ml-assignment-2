#include <math.h>
#include <chrono>

#include <fmt/core.h>

#include "neuralnet.h"

/* NOTE TO SELF ON CMAKE:
 *  Add new .cpp files to the cmake file and then rebuild the make file
*/

int main() {

    nnlayer_t test_layer = init_layer(0, 5, 2);

    std::vector<size_t> test_structure{1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000};
    nn_t test_network = init_network(test_structure);

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

    Eigen::VectorXd input(1000);
    input = Eigen::VectorXd::Constant(1000, 1);

    auto start_time = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd fprop_result = test_network.eval_network(input, activation_relu);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> execute_time = end_time - start_time;

    fmt::println("Execution Time: {} ms", execute_time.count());
    fmt::println("Neural Network Memory size: {}", sizeof(test_network));

    return 0;
}