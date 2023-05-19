#include <math.h>

#include <fmt/core.h>

#include "neuralnet.h"

/* NOTE TO SELF ON CMAKE:
 *  Add new .cpp files to the cmake file and then rebuild the make file
*/

int main() {

    nnlayer_t test_layer = init_layer(5, 2);
    test_layer.print();

    auto activation_LU = [](double x) { return x; };
    auto activation_S = [](double x) { return 1.0 / (1.0 + std::exp(-1.0 * x)); };
    auto activation_ReLU = [](double x) { return x > 0 ? x : 0; };

    Eigen::VectorXd input {{2.0, 3.0, 4.0, 5.0, 6.0}};
    Eigen::VectorXd result_lu = test_layer.fprop_layer(input, activation_LU);
    Eigen::VectorXd result_s = test_layer.fprop_layer(input, activation_S);
    Eigen::VectorXd result_relu = test_layer.fprop_layer(input, activation_ReLU);

    fmt::print("Input Vector:\n{}\n", input);
    fmt::print("Linear Unit Result Vector:\n{}\n", result_lu);
    fmt::print("Sigmoid Unit Result Vector:\n{}\n", result_s);
    fmt::print("ReLU Result Vector:\n{}\n", result_relu);

    return 0;
}