#include <math.h>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <fstream>

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
    fmt::print("\x1B[2J\x1b[H"); // Clear the terminal
    nnspec_t& spec = all_specs[selected_spec];

    spec.print_spec_info();

    // Load spec data
    std::ifstream data_file(spec.data.data_file.string());
    std::ifstream label_file(spec.data.label_file.string());

    Eigen::MatrixXd data(spec.data.size, spec.data.feature_count);
    Eigen::VectorXd labels(spec.data.size);

    if (data_file.is_open()) {
        for (size_t i = 0; i < spec.data.size; ++i) {
            fmt::print(fg(fmt::color::blue), "\rLoading: "); fmt::print("point {} of {}", i + 1, spec.data.size);
            std::string line;
            getline(data_file, line);
            Eigen::VectorXd data_point(spec.data.feature_count);
            std::vector<std::string> split_strs;
            for (size_t j = 0; j < spec.data.feature_count; ++j) {
                data_point(j) = std::stod(line.substr(0, line.find(",")));
                line.erase(0, line.find(",") + 1);
            }

            data.row(i) = data_point;
        }
        fmt::print(fg(fmt::color::blue), "\rLoaded data points                                        \n");
    } else {
        fmt::print(fg(fmt::color::red), "Failed to open the file {}. Please make sure the path is correct.\n", spec.data.data_file.string());
    }

    if (label_file.is_open()) {
        for (size_t i = 0; i < spec.data.size; ++i) {
            fmt::print(fg(fmt::color::blue), "\rLoading: "); fmt::print("label {} of {}", i + 1, spec.data.size);
            std::string line;
            getline(label_file, line);
            labels(i) = std::stoi(line);
        }
        fmt::print(fg(fmt::color::blue), "\rLoaded labels                                                   \n");
    } else {
        fmt::print(fg(fmt::color::red), "Failed to open the file {}. Please make sure the path is correct.\n", spec.data.label_file.string());
    }


    data_file.close();
    label_file.close();

    std::vector<size_t> structure{71, 10};
    nn_t network = init_network(structure);

    auto activation_sig = [](const Eigen::VectorXd& values) { return values.unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); }); };
    fmt::println("{}", network.eval_network_perf(spec.data.label_count, data, labels, activation_sig));

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