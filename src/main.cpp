#include <math.h>
#include <chrono>
#include <filesystem>
#include <iostream>

#include <fmt/core.h>
#include <fmt/color.h>
#include <toml++/toml.h>

#include "neuralnet.h"
#include "parser.h"

/* NOTE TO SELF ON CMAKE:
 *  Add new .cpp files to the cmake file and then rebuild the make file
*/

void execute_training();
void test_network();

int main() {
    execute_training();
    return 0;
}

void execute_training() {
    const std::filesystem::path network_spec_dir(std::filesystem::current_path()/"data/training_specs");
    std::vector<nnspec_t> all_specs;

    fmt::println("Available Network Specifications");
    for (const auto& entry: std::filesystem::directory_iterator(network_spec_dir)) {
        nnspec_t spec = load_network_spec_from_file(entry);
        all_specs.push_back(spec);
        fmt::println("\t{}: {}", all_specs.size(), spec.name);
    }

    fmt::print("Type the number of the network specification to use: ");
    size_t selected_spec = -1;
    std::cin >> selected_spec;
    --selected_spec; //Change the typed number 1 -> n to an index of 0 -> n-1
    fmt::println("");
    nnspec_t& spec = all_specs[selected_spec];

    // Load spec data
    FILE* data_file = fopen(spec.data.data_file.string().c_str(), "r");
    FILE* label_file = fopen(spec.data.label_file.string().c_str(), "r");

    { // File IO Error handling
        if (data_file == nullptr) {
            fmt::print(fg(fmt::color::red), "Failed to open the file {}. Please make sure the path is correct.\n", spec.data.data_file.string());
            return;
        }

        if (label_file == nullptr) {
            fmt::print(fg(fmt::color::red), "Failed to open the file {}. Please make sure the path is correct.\n", spec.data.label_file.string());
            return;
        }
    }

    // Current Network Spec Info
    spec.print_spec_info();

    fclose(data_file);
    fclose(label_file);
}

void test_network() {
    nnlayer_t test_layer = init_layer(0, 5, 2);

    std::vector<size_t> test_structure{1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000};
    nn_t test_network = init_network(test_structure);

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
}