#include <math.h>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <stdlib.h>

#include <fmt/core.h>
#include <fmt/color.h>
#include <toml++/toml.h>

#include "network.h"
#include "spec.h"

/* NOTE TO SELF ON CMAKE:
 *  Add new .cpp files to the cmake file and then rebuild the make file
*/


void execute_training();
void test_network();
void test_refactor();

int main() {
    srand(static_cast<uint32_t>(time(0)));
    test_refactor();
    return 0;
}

void test_refactor() {
    const std::filesystem::path network_spec_dir(std::filesystem::current_path()/"data/training_specs");
    std::vector<NeuralNetworkSpecification> all_specs;

    fmt::print(fg(fmt::terminal_color::yellow), "Available Network Specifications\n");
    for (const auto& entry: std::filesystem::directory_iterator(network_spec_dir)) {
        toml::table spec_file = toml::parse_file(entry.path().string());
        all_specs.push_back(NeuralNetworkSpecification(spec_file));
        fmt::print("\t| {:<3} | id: {:<9} | name: {:<32} |\n", 
            fmt::format(fg(fmt::terminal_color::blue), "{}", all_specs.size()), 
            fmt::format(fg(fmt::terminal_color::blue), "{}", all_specs.back().id.substr(0, 7)),
            fmt::format(fg(fmt::terminal_color::green), "{}", all_specs.back().name)
        );
    }

    fmt::print(fg(fmt::terminal_color::magenta), "\nType the number of the network specification to use: ");
    size_t selected_spec = -1;
    std::cin >> selected_spec;
    --selected_spec; //Change the typed number 1 -> n to an index of 0 -> n-1
    NeuralNetworkSpecification& spec = all_specs[selected_spec];

    // Load spec data
    std::ifstream data_file(spec.data_file.string());
    std::ifstream label_file(spec.label_file.string());

    Eigen::MatrixXd data(spec.data_size, spec.num_features);
    Eigen::VectorXd labels(spec.data_size);

    fmt::println("");
    if (data_file.is_open()) {
        for (size_t i = 0; i < spec.data_size; ++i) {
            fmt::print(fg(fmt::terminal_color::cyan), "\rLoading: "); fmt::print("point {} of {}", i + 1, spec.data_size);
            std::string line;
            getline(data_file, line);
            Eigen::VectorXd data_point(spec.num_features);
            std::vector<std::string> split_strs;
            for (size_t j = 0; j < spec.num_features; ++j) {
                data_point(j) = std::stod(line.substr(0, line.find(",")));
                line.erase(0, line.find(",") + 1);
            }

            data.row(i) = data_point;
        }
        fmt::print(fg(fmt::terminal_color::cyan), "\rLoaded data points                                        \n");
    } else {
        // Error Handling
        fmt::print(fg(fmt::terminal_color::bright_red), "Failed to open data file {}\n", spec.data_file.string());
        return;
    }

    if (label_file.is_open()) {
        for (size_t i = 0; i < spec.data_size; ++i) {
            fmt::print(fg(fmt::terminal_color::cyan), "\rLoading: "); fmt::print("label {} of {}", i + 1, spec.data_size);
            std::string line;
            getline(label_file, line);
            labels(i) = std::stoi(line);
        }
        fmt::print(fg(fmt::terminal_color::cyan), "\rLoaded labels                                                   \n");
    } else {
        // Error handling
        fmt::print(fg(fmt::terminal_color::bright_red), "Failed to open label file {}\n", spec.label_file.string());
        return;
    }
    fmt::println("");

    data_file.close();
    label_file.close();

    spec.create_networks();
    spec.train_networks(data, labels);
}